//go:build windows

package supervisor

import (
	"fmt"
	"runtime"
	"sync"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

type SingleInstance struct {
	stop       windows.Handle
	done       chan struct{}
	once       sync.Once
	ShowWindow <-chan struct{}
	Abandoned  bool
}

var instanceSuffix string

func UserSID() (string, error) {
	token, err := windows.OpenCurrentProcessToken()
	if err != nil {
		return "", err
	}
	defer token.Close()
	user, err := token.GetTokenUser()
	if err != nil {
		return "", err
	}
	return user.User.Sid.String(), nil
}

// AcquireSingleInstance returns primary=false after signaling the existing
// window. Keep the primary lease until Supervisor.Exit has completed.
func AcquireSingleInstance() (*SingleInstance, bool, error) {
	sid, err := UserSID()
	if err != nil {
		return nil, false, err
	}
	return acquireSingleInstance(sid, instanceSuffix)
}
func acquireSingleInstance(sid, suffix string) (*SingleInstance, bool, error) {
	name := "Local\\Bubo.Control." + sid + suffix
	pipe := `\\.\pipe\Bubo.Control.` + sid + suffix
	sd, err := windows.SecurityDescriptorFromString("D:P(A;;GA;;;" + sid + ")")
	if err != nil {
		return nil, false, err
	}
	sa := windows.SecurityAttributes{Length: uint32(unsafe.Sizeof(windows.SecurityAttributes{})), SecurityDescriptor: sd}
	type answer struct {
		i       *SingleInstance
		primary bool
		err     error
	}
	result := make(chan answer, 1)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		n, _ := windows.UTF16PtrFromString(name)
		mutex, err := windows.CreateMutex(&sa, false, n)
		if err != nil && err != windows.ERROR_ALREADY_EXISTS {
			result <- answer{err: err}
			return
		}
		defer windows.CloseHandle(mutex)
		state, err := windows.WaitForSingleObject(mutex, 0)
		if err != nil {
			result <- answer{err: err}
			return
		}
		if state == uint32(windows.WAIT_TIMEOUT) {
			result <- answer{primary: false, err: activate(pipe)}
			return
		}
		if state != windows.WAIT_OBJECT_0 && state != windows.WAIT_ABANDONED {
			result <- answer{err: fmt.Errorf("mutex wait %d", state)}
			return
		}
		defer windows.ReleaseMutex(mutex)
		stop, err := windows.CreateEvent(nil, 1, 0, nil)
		if err != nil {
			result <- answer{err: err}
			return
		}
		defer windows.CloseHandle(stop)
		show := make(chan struct{}, 1)
		i := &SingleInstance{stop: stop, done: make(chan struct{}), ShowWindow: show, Abandoned: state == windows.WAIT_ABANDONED}
		// Make the first pipe before publishing the lease to close the launch race.
		h, err := newActivationPipe(pipe, &sa)
		if err != nil {
			result <- answer{err: err}
			return
		}
		result <- answer{i, true, nil}
		defer close(i.done)
		defer close(show)
		for {
			event, _ := windows.CreateEvent(nil, 1, 0, nil)
			ov := windows.Overlapped{HEvent: event}
			err = windows.ConnectNamedPipe(h, &ov)
			connected := err == nil || err == windows.ERROR_PIPE_CONNECTED
			if err == windows.ERROR_IO_PENDING {
				wait, e := windows.WaitForMultipleObjects([]windows.Handle{stop, event}, false, windows.INFINITE)
				connected = e == nil && wait == windows.WAIT_OBJECT_0+1
			} else if !connected {
				windows.CloseHandle(event)
				windows.CloseHandle(h)
				return
			}
			if !connected {
				windows.CancelIoEx(h, &ov)
				var n uint32
				_ = windows.GetOverlappedResult(h, &ov, &n, true)
				windows.CloseHandle(event)
				windows.CloseHandle(h)
				return
			}
			// An accepted connection carries only activation; no commands or secrets.
			select {
			case show <- struct{}{}:
			default:
			}
			windows.DisconnectNamedPipe(h)
			windows.CloseHandle(h)
			windows.CloseHandle(event)
			state, _ = windows.WaitForSingleObject(stop, 0)
			if state == windows.WAIT_OBJECT_0 {
				return
			}
			h, err = newActivationPipe(pipe, &sa)
			if err != nil {
				return
			}
		}
	}()
	a := <-result
	return a.i, a.primary, a.err
}
func newActivationPipe(name string, sa *windows.SecurityAttributes) (windows.Handle, error) {
	n, _ := windows.UTF16PtrFromString(name)
	return windows.CreateNamedPipe(n, windows.PIPE_ACCESS_DUPLEX|windows.FILE_FLAG_OVERLAPPED|windows.FILE_FLAG_FIRST_PIPE_INSTANCE, windows.PIPE_TYPE_BYTE|windows.PIPE_READMODE_BYTE|windows.PIPE_REJECT_REMOTE_CLIENTS, 1, 128, 128, 0, sa)
}
func activate(name string) error {
	n, _ := windows.UTF16PtrFromString(name)
	deadline := time.Now().Add(2 * time.Second)
	for {
		h, err := windows.CreateFile(n, windows.GENERIC_READ|windows.GENERIC_WRITE, 0, nil, windows.OPEN_EXISTING, 0, 0)
		if err == nil {
			windows.CloseHandle(h)
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("Show window activation failed: %w", err)
		}
		time.Sleep(10 * time.Millisecond)
	}
}
func (i *SingleInstance) Close() { i.once.Do(func() { windows.SetEvent(i.stop); <-i.done }) }
