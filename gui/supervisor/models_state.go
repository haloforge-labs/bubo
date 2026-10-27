package supervisor

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"
	"unicode"
)

// What the window reads under the frame key `models`. Rows come from the CLI;
// the few values the CLI does not give (free disk space of the target drive,
// the folders other programs keep models in) are plain operating-system reads.

const (
	// ModelsLoading: the first list has not arrived yet.
	ModelsLoading = "loading"
	// ModelsOK: the list below is the CLI's current answer.
	ModelsOK = "ok"
	// ModelsNotReported: the list could not be read. Reason says why. The
	// lists are null; the window says "not reported" and shows the reason.
	ModelsNotReported = "not_reported"
)

// ModelsDisk is the free space of the drive that receives downloads. FreeBytes
// is null when the drive could not be read (Error says why).
type ModelsDisk struct {
	Drive     string  `json:"drive"`
	Path      string  `json:"path"`
	FreeBytes *uint64 `json:"free_bytes"`
	Error     string  `json:"error,omitempty"`
}

// SuggestedFolder is a folder other programs keep models in. Only folders
// that exist are listed. Nothing is scanned until the user adds the folder.
type SuggestedFolder struct {
	Label  string `json:"label"`
	Path   string `json:"path"`
	Exists bool   `json:"exists"`
	Added  bool   `json:"added"`
}

// ModelsState is the frame key `models`.
type ModelsState struct {
	Status     string `json:"status"`
	Reason     string `json:"reason,omitempty"`
	ReasonCode string `json:"reason_code,omitempty"`
	// ExitCode is the CLI exit code behind a failed read (0 when none).
	ExitCode  int    `json:"exit_code,omitempty"`
	UpdatedAt string `json:"updated_at,omitempty"`
	// Refreshing is true while a list read is running.
	Refreshing       bool              `json:"refreshing"`
	Listing          *ModelsListing    `json:"listing"`
	Catalogue        []CatalogueEntry  `json:"catalogue"`
	PreparedModels   []PreparedModel   `json:"prepared_models"`
	Scanned          []ScannedModel    `json:"scanned"`
	ScanWarnings     []ScanWarning     `json:"scan_warnings"`
	Disk             ModelsDisk        `json:"disk"`
	SuggestedFolders []SuggestedFolder `json:"suggested_folders"`
	// Folders are the folders the user added (saved in the settings file).
	Folders  []string  `json:"folders"`
	Download *ModelJob `json:"download"`
}

// ModelJobError is a failed download, repair or check, as the CLI reported it.
type ModelJobError struct {
	Code     string `json:"code"`
	Remedy   string `json:"remedy"`
	ExitCode int    `json:"exit_code"`
	Detail   string `json:"detail,omitempty"`
}

// ModelJob is the one download, repair or check the window may run at a time.
type ModelJob struct {
	ID   string `json:"id"`
	Kind string `json:"kind"` // download, repair, check
	// State: planning, awaiting_confirm, running, cancelling, done, failed, cancelled.
	State        string        `json:"state"`
	Phase        string        `json:"phase,omitempty"`
	FileID       string        `json:"file_id,omitempty"`
	FilePosition *FilePosition `json:"file_position,omitempty"`
	// Overall is the whole job (every planned file). BytesDone and BytesTotal
	// below describe only the current file.
	Overall   *JobProgress `json:"overall,omitempty"`
	BytesDone uint64       `json:"bytes_done"`
	// BytesTotal is null while the CLI does not know the size.
	BytesTotal *uint64 `json:"bytes_total"`
	// Stalled is true when no progress arrived for StallSeconds. This only
	// warns. The window offers Cancel; nothing stops by itself.
	Stalled      bool           `json:"stalled"`
	StallSeconds int            `json:"stall_seconds,omitempty"`
	Message      string         `json:"message,omitempty"`
	Error        *ModelJobError `json:"error"`
	Plan         *ModelPlan     `json:"plan"`
	Prepared     bool           `json:"prepared"`
	ManifestPath *string        `json:"manifest_path"`
	// RepairOutput is the --output of a repair or check that was started for one
	// package (the row's repair_output), so a retry acts on the same package.
	RepairOutput string `json:"repair_output,omitempty"`
	UpdatedAt    string `json:"updated_at"`
}

// FilePosition is a catalogue ordinal resolved through an explicit CLI plan
// file identity. Unknown identities never imply a whole-model total.
type FilePosition struct {
	Index int `json:"index"`
	Total int `json:"total"`
}

func catalogueFilePosition(fileID string, plan *ModelPlan, catalogue []CatalogueEntry) *FilePosition {
	if fileID == "" || plan == nil {
		return nil
	}
	var result *FilePosition
	var matchedID string
	for _, source := range plan.Files {
		if source.FileID != fileID {
			continue
		}
		for _, entry := range catalogue {
			if plan.ID != "" && entry.ID != plan.ID {
				continue
			}
			for i, file := range entry.Files {
				if (plan.ID != "" && file.Name == filepath.Base(source.Path)) || (source.URL != "" && source.URL == file.URL) {
					if result != nil && matchedID != entry.ID {
						return nil // A shared file URL does not identify a model.
					}
					result, matchedID = &FilePosition{Index: i + 1, Total: len(entry.Files)}, entry.ID
				}
			}
		}
	}
	return result
}

// JobProgress is the whole download job: the planned bytes of every file that
// still has to transfer, and how many of them are done. The CLI reports bytes
// for the current file only, so the supervisor adds the files before it.
type JobProgress struct {
	BytesDone  uint64 `json:"bytes_done"`
	BytesTotal uint64 `json:"bytes_total"`
}

// jobFile is one planned file, with the component that owns it.
type jobFile struct {
	Component string
	FileID    string
	Size      uint64
	State     string
}

func planJobFiles(plan ModelPlan) []jobFile {
	var files []jobFile
	for _, f := range plan.Files {
		files = append(files, jobFile{Component: plan.Kind, FileID: f.FileID, Size: f.Size, State: f.State})
	}
	return files
}

// partJobFiles lists the files of the plan records a prepare run sent, the
// engine files first, each component in part order. It reads no licence text,
// so it is cheap enough for every progress record.
func partJobFiles(parts []Plan) []jobFile {
	ordered := append([]Plan(nil), parts...)
	sort.SliceStable(ordered, func(i, j int) bool {
		a, b := ordered[i], ordered[j]
		if (a.Component == "runtime") != (b.Component == "runtime") {
			return a.Component == "runtime"
		}
		if a.Component != b.Component {
			return a.Component < b.Component
		}
		return a.Part < b.Part
	})
	var files []jobFile
	for _, p := range ordered {
		for _, source := range p.Sources {
			files = append(files, jobFile{Component: p.Component, FileID: source.FileID, Size: source.Size})
		}
	}
	// Each component's header lists one state per source, in source order;
	// the other records repeat that list.
	for _, p := range ordered {
		if p.Part != 0 {
			continue
		}
		var own []int
		for i, f := range files {
			if f.Component == p.Component {
				own = append(own, i)
			}
		}
		if len(p.FileStates) == len(own) {
			for k, i := range own {
				files[i].State = p.FileStates[k]
			}
		}
	}
	return files
}

// jobProgress adds up the files in order (the CLI works through each plan's
// files in the order it lists them). A file the CLI reuses is not counted.
// It returns nil when the current file is not in the plan, for example while
// the CLI unpacks the engine files, unless the phase shows that every
// download has ended.
func jobProgress(files []jobFile, component, fileID, phase string, done uint64) *JobProgress {
	var total, before uint64
	found := false
	for _, f := range files {
		if f.State == "reuse" {
			continue
		}
		total += f.Size
		switch {
		case found:
		case f.Component == component && f.FileID == fileID:
			found = true
			before += min(done, f.Size)
		default:
			before += f.Size
		}
	}
	if total == 0 {
		return nil
	}
	if !found {
		if phase == "extract" || phase == "promote" {
			return &JobProgress{BytesDone: total, BytesTotal: total}
		}
		return nil
	}
	return &JobProgress{BytesDone: min(before, total), BytesTotal: total}
}

// OverallProgress is the whole-job number the supervisor adds to a progress
// event from the plan parts it holds: the same rule as the running flow.
func OverallProgress(parts []Plan, p *Progress) *JobProgress {
	return jobProgress(partJobFiles(parts), p.Component, p.FileID, p.Phase, p.Done)
}

// ModelPlanFile is one file of the plan (a source row of `prepare --plan`).
type ModelPlanFile struct {
	FileID     string `json:"file_id"`
	URL        string `json:"url"`
	Size       uint64 `json:"size"`
	SHA256     string `json:"sha256"`
	Repository string `json:"repository"`
	Revision   string `json:"revision"`
	Path       string `json:"path"`
	LicenseID  string `json:"license_id"`
	// State is download, reuse, import or resume when the CLI said.
	State string `json:"state,omitempty"`
	// Replaces is the CLI's reason for replacing a file in the package folder.
	Replaces string `json:"replaces"`
}

// ModelPlanLicence is the full licence text of the plan, joined from the
// CLI's chunks. The window shows it before the first download.
type ModelPlanLicence struct {
	Path   string `json:"path"`
	Text   string `json:"text"`
	SHA256 string `json:"sha256"`
}

// ModelPlan is what `prepare model --catalogue-id ID --plan` returned. Nothing
// is written and nothing is downloaded until ConfirmModelDownload.
type ModelPlan struct {
	ID       string             `json:"id"`
	Kind     string             `json:"kind"`
	Digest   string             `json:"plan_digest"`
	Ready    bool               `json:"ready"`
	Files    []ModelPlanFile    `json:"files"`
	Licences []ModelPlanLicence `json:"licences"`
	Paths    map[string]string  `json:"paths"`
	Volumes  []Volume           `json:"volumes"`
	// RequiredBytes and FreeBytes are the numbers of the tightest volume.
	// Drive names it. EnoughSpace is false when any volume is short.
	RequiredBytes uint64 `json:"required_bytes"`
	FreeBytes     uint64 `json:"free_bytes"`
	Drive         string `json:"drive"`
	EnoughSpace   bool   `json:"enough_space"`
	// Licence is the catalogue licence block (the exact prompt line).
	Licence *ModelLicence `json:"licence"`
}

// assemblePlan joins the plan records of one component into a ModelPlan. A
// missing, repeated or disagreeing part is a Protocol error.
func assemblePlan(parts []Plan, component string) (ModelPlan, error) {
	var out ModelPlan
	var total uint64
	seen := map[uint64]bool{}
	type chunk struct {
		offset uint64
		text   string
	}
	chunks := map[string][]chunk{}
	hashes := map[string]string{}
	sizes := map[string]uint64{}
	var order []string
	var states []string
	var replaces []string
	// Files follow the part order, so the header's states pair with them.
	ordered := append([]Plan(nil), parts...)
	sort.SliceStable(ordered, func(i, j int) bool { return ordered[i].Part < ordered[j].Part })
	for _, p := range ordered {
		if p.Component != component {
			continue
		}
		if total == 0 {
			total, out.Digest = p.Parts, p.Digest
		}
		if p.Parts != total || p.Digest != out.Digest || p.Part >= total || seen[p.Part] {
			return ModelPlan{}, protocol("the plan parts disagree or repeat")
		}
		seen[p.Part] = true
		if p.Part == 0 {
			out.Ready = p.Ready // the header record says whether the files are ready
			// One state per source, in source order. The other records repeat
			// the whole list, so only the header's copy is read.
			states = p.FileStates
			replaces = p.Replaces
		} else if (p.Replaces == nil) != (replaces == nil) || !slices.Equal(p.Replaces, replaces) {
			return ModelPlan{}, protocol("the plan replacement lists disagree")
		}
		out.Files = append(out.Files, func() []ModelPlanFile {
			var rows []ModelPlanFile
			for _, s := range p.Sources {
				rows = append(rows, ModelPlanFile{FileID: s.FileID, URL: s.URL, Size: s.Size, SHA256: s.SHA256, Repository: s.Repository, Revision: s.Revision, Path: s.Path, LicenseID: s.LicenseID})
			}
			return rows
		}()...)
		for _, l := range p.Licenses {
			if _, known := hashes[l.Path]; !known {
				order = append(order, l.Path)
				hashes[l.Path] = l.SHA256
				sizes[l.Path] = l.TextBytes
			} else if hashes[l.Path] != l.SHA256 || sizes[l.Path] != l.TextBytes {
				return ModelPlan{}, protocol("licence chunks of " + l.Path + " disagree")
			}
			chunks[l.Path] = append(chunks[l.Path], chunk{l.TextOffset, l.Text})
		}
		if len(p.Paths) > 0 {
			if out.Paths == nil {
				out.Paths = map[string]string{}
			}
			for k, raw := range p.Paths {
				var s string
				if err := jsonString(raw, &s); err != nil {
					return ModelPlan{}, protocol("plan path " + k + " is not a string")
				}
				out.Paths[k] = s
			}
		}
		out.Volumes = append(out.Volumes, p.Volumes...)
	}
	if total == 0 || uint64(len(seen)) != total {
		return ModelPlan{}, protocol("the plan is incomplete")
	}
	if len(states) == len(out.Files) {
		for i := range out.Files {
			out.Files[i].State = states[i]
		}
	}
	if replaces != nil {
		if len(replaces) != len(out.Files) {
			return ModelPlan{}, protocol("the plan replacement list does not match its sources")
		}
		for i := range out.Files {
			out.Files[i].Replaces = replaces[i]
		}
	}
	for _, path := range order {
		cs := chunks[path]
		sort.Slice(cs, func(i, j int) bool { return cs[i].offset < cs[j].offset })
		var sb strings.Builder
		var next uint64
		for _, c := range cs {
			if c.offset != next {
				return ModelPlan{}, protocol("licence text of " + path + " has a gap or an overlap")
			}
			sb.WriteString(c.text)
			next += uint64(len([]byte(c.text)))
		}
		if next != sizes[path] {
			return ModelPlan{}, protocol("licence text of " + path + " is incomplete")
		}
		out.Licences = append(out.Licences, ModelPlanLicence{Path: path, Text: sb.String(), SHA256: hashes[path]})
	}
	out.EnoughSpace = true
	var tight int64 = 1<<63 - 1
	for _, v := range out.Volumes {
		margin := int64(v.FreeBytes) - int64(v.RequiredBytes)
		if v.FreeBytes < v.RequiredBytes {
			out.EnoughSpace = false
		}
		if margin < tight {
			tight = margin
			out.RequiredBytes, out.FreeBytes, out.Drive = v.RequiredBytes, v.FreeBytes, v.Name
		}
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// Folder and subject checks. Plain string and path rules; nothing is scanned.

const MaxModelFolders = 32

// NormalizeModelFolderPath checks the text of a folder the user chose and
// returns its clean form. It does not touch the disk: the CLI refuses drive
// roots, links and missing folders itself and reports them as scan warnings,
// and the caller checks existence for a friendlier message.
func NormalizeModelFolderPath(p string) (string, error) {
	p = strings.TrimSpace(p)
	if p == "" {
		return "", fmt.Errorf("Choose a folder.")
	}
	for _, r := range p {
		if unicode.IsControl(r) {
			return "", fmt.Errorf("The folder name has a control character.")
		}
	}
	if strings.HasPrefix(p, "-") {
		return "", fmt.Errorf("The folder name cannot start with a hyphen.")
	}
	if !filepath.IsAbs(p) {
		return "", fmt.Errorf("Choose a full folder path, for example D:\\Models.")
	}
	c := filepath.Clean(p)
	vol := filepath.VolumeName(c)
	rest := strings.Trim(strings.TrimPrefix(c, vol), `\/`)
	if rest == "" {
		return "", fmt.Errorf("Choose a folder, not a whole drive.")
	}
	if strings.HasPrefix(vol, `\\`) && len(strings.Split(strings.Trim(c, `\`), `\`)) <= 2 {
		return "", fmt.Errorf("Choose a folder inside the share, not the share itself.")
	}
	return c, nil
}

var catalogueIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`)

// ValidCatalogueID is true for a catalogue id as the CLI writes it.
func ValidCatalogueID(id string) bool { return catalogueIDPattern.MatchString(id) }

// SameFolder compares two folder paths the way Windows does.
func SameFolder(a, b string) bool {
	return strings.EqualFold(strings.TrimRight(filepath.Clean(a), `\/`), strings.TrimRight(filepath.Clean(b), `\/`))
}

func jsonString(raw json.RawMessage, out *string) error { return json.Unmarshal(raw, out) }
