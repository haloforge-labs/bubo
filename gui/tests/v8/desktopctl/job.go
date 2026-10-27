//go:build windows

package main

import (
	"encoding/binary"
	"fmt"
	"path/filepath"
	"sort"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

type jobProcess struct {
	PID          uint32 `json:"pid"`
	CreationTime string `json:"creation_time"`
	Image        string `json:"image"`
}

type trackedProcess struct {
	jobProcess
	handle windows.Handle
}

func createTestJob() (windows.Handle, error) {
	job, err := windows.CreateJobObject(nil, nil)
	if err != nil {
		return 0, fmt.Errorf("CreateJobObject: %w", err)
	}
	var limits windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION
	limits.BasicLimitInformation.LimitFlags = windows.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
	_, err = windows.SetInformationJobObject(job, windows.JobObjectExtendedLimitInformation,
		uintptr(unsafe.Pointer(&limits)), uint32(unsafe.Sizeof(limits)))
	if err != nil {
		windows.CloseHandle(job)
		return 0, fmt.Errorf("SetInformationJobObject(KILL_ON_JOB_CLOSE): %w", err)
	}
	var actual windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION
	if err := windows.QueryInformationJobObject(job, windows.JobObjectExtendedLimitInformation,
		uintptr(unsafe.Pointer(&actual)), uint32(unsafe.Sizeof(actual)), nil); err != nil {
		windows.CloseHandle(job)
		return 0, fmt.Errorf("QueryInformationJobObject(limits): %w", err)
	}
	if actual.BasicLimitInformation.LimitFlags != windows.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE {
		windows.CloseHandle(job)
		return 0, fmt.Errorf("job limits changed: got 0x%X, expected only KILL_ON_JOB_CLOSE", actual.BasicLimitInformation.LimitFlags)
	}
	return job, nil
}

// jobPIDs reads only the kernel's current job membership. A short buffer may
// return a partial list, so a snapshot is never accepted until it is complete.
func jobPIDs(job windows.Handle) ([]uint32, error) {
	const maxMembers = 16384
	wordSize := int(unsafe.Sizeof(uintptr(0)))
	capacity := 32
	for capacity <= maxMembers {
		buf := make([]byte, 8+capacity*wordSize)
		err := windows.QueryInformationJobObject(job, windows.JobObjectBasicProcessIdList,
			uintptr(unsafe.Pointer(&buf[0])), uint32(len(buf)), nil)
		assigned := binary.LittleEndian.Uint32(buf[0:4])
		listed := binary.LittleEndian.Uint32(buf[4:8])
		if err == windows.ERROR_MORE_DATA || (err == nil && listed < assigned) {
			capacity *= 2
			if int(assigned) > capacity {
				capacity = int(assigned)
			}
			continue
		}
		if err != nil {
			return nil, fmt.Errorf("QueryInformationJobObject(process IDs): %w", err)
		}
		if int(listed) > capacity {
			return nil, fmt.Errorf("job PID count %d exceeds buffer capacity %d", listed, capacity)
		}
		pids := make([]uint32, 0, listed)
		for i := 0; i < int(listed); i++ {
			offset := 8 + i*wordSize
			var value uint64
			if wordSize == 8 {
				value = binary.LittleEndian.Uint64(buf[offset : offset+8])
			} else {
				value = uint64(binary.LittleEndian.Uint32(buf[offset : offset+4]))
			}
			if value > uint64(^uint32(0)) {
				return nil, fmt.Errorf("job PID out of range: %d", value)
			}
			pids = append(pids, uint32(value))
		}
		sort.Slice(pids, func(i, j int) bool { return pids[i] < pids[j] })
		return pids, nil
	}
	return nil, fmt.Errorf("job PID list exceeded %d entries", maxMembers)
}

func processCreationTime(handle windows.Handle) (string, error) {
	var creation, exit, kernel, user windows.Filetime
	if err := windows.GetProcessTimes(handle, &creation, &exit, &kernel, &user); err != nil {
		return "", err
	}
	return fmt.Sprintf("%08X%08X", creation.HighDateTime, creation.LowDateTime), nil
}

func processImage(handle windows.Handle) string {
	buf := make([]uint16, 32768)
	size := uint32(len(buf))
	if err := windows.QueryFullProcessImageName(handle, 0, &buf[0], &size); err != nil {
		return "unknown"
	}
	return filepath.Base(syscall.UTF16ToString(buf[:size]))
}

// refreshJobMembers opens only PIDs returned by this job. Open handles retain
// the original process identity after exit and cannot turn into a reused PID.
func refreshJobMembers(job windows.Handle, tracked map[string]trackedProcess) (map[uint32]jobProcess, error) {
	pids, err := jobPIDs(job)
	if err != nil {
		return nil, err
	}
	members := make(map[uint32]jobProcess, len(pids))
	for _, pid := range pids {
		handle, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION|windows.SYNCHRONIZE, false, pid)
		if err == windows.ERROR_INVALID_PARAMETER { // Exited after the PID list was read.
			continue
		}
		if err != nil {
			return nil, fmt.Errorf("OpenProcess(job PID %d): %w", pid, err)
		}
		created, err := processCreationTime(handle)
		if err != nil {
			windows.CloseHandle(handle)
			return nil, fmt.Errorf("GetProcessTimes(job PID %d): %w", pid, err)
		}
		key := fmt.Sprintf("%d/%s", pid, created)
		member, exists := tracked[key]
		if exists {
			windows.CloseHandle(handle)
		} else {
			member = trackedProcess{jobProcess: jobProcess{PID: pid, CreationTime: created, Image: processImage(handle)}, handle: handle}
			tracked[key] = member
		}
		members[pid] = member.jobProcess
	}
	return members, nil
}

func jobExits(tracked map[string]trackedProcess) ([]processExit, error) {
	result := make([]processExit, 0, len(tracked))
	for _, member := range tracked {
		var code uint32
		if err := windows.GetExitCodeProcess(member.handle, &code); err != nil {
			return nil, fmt.Errorf("GetExitCodeProcess(job PID %d): %w", member.PID, err)
		}
		if code != 259 { // STILL_ACTIVE
			result = append(result, processExit{member.PID, member.CreationTime, member.Image, code, fmt.Sprintf("0x%08X", code)})
		}
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i].PID == result[j].PID {
			return result[i].CreationTime < result[j].CreationTime
		}
		return result[i].PID < result[j].PID
	})
	return result, nil
}

func closeTracked(tracked map[string]trackedProcess) {
	for _, member := range tracked {
		windows.CloseHandle(member.handle)
	}
}
