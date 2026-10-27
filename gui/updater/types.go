package updater

import (
	"crypto/x509"
	"fmt"
	"time"
)

const ManifestURL = "https://github.com/haloforge-labs/bubo/releases/latest/download/bubo-release.json"

const (
	DefaultHealthTimeout = 90 * time.Second
	DefaultWaitTimeout   = 60 * time.Second
)

// The U4 GUI health handshake is deliberately independent of server/model
// startup. The helper passes HealthFlag followed by a random token, and
// HealthDeadlineFlag followed by a UTC RFC3339Nano deadline. After initialization
// succeeds, the GUI atomically writes HealthContent(token) to HealthFileName
// inside its install folder. It must exit on initialization failure or by the
// deadline if it cannot acknowledge health. A stale token never counts as health.
// The helper never kills a GUI that fails this contract: rollback remains pending
// until that exact process exits and Windows releases its image.
const (
	HealthFlag         = "--bubo-update-health"
	HealthDeadlineFlag = "--bubo-update-health-deadline"
	HealthFileName     = ".bubo-update-health"
	ResultFileName     = ".bubo-update-result.json"
)

func HealthContent(token string) []byte { return []byte("bubo-update-health-v1 " + token + "\n") }

type Code string

const (
	CodeConfig               Code = "configuration"
	CodeURL                  Code = "url_refused"
	CodeNetwork              Code = "network"
	CodeUpdateInfoMissing    Code = "update_information_missing"
	CodeUpdateServer         Code = "update_server_error"
	CodeManifest             Code = "invalid_manifest"
	CodeUnsupported          Code = "unsupported"
	CodeDowngrade            Code = "downgrade_refused"
	CodeStale                Code = "stale_manifest"
	CodeIntermediateRequired Code = "intermediate_required"
	CodeNotChecked           Code = "check_required"
	CodeBusy                 Code = "busy"
	CodeSpace                Code = "insufficient_space"
	CodeIO                   Code = "filesystem"
	CodeSize                 Code = "size_mismatch"
	CodeHash                 Code = "sha256_mismatch"
	CodeResume               Code = "invalid_resume"
	CodeCanceled             Code = "canceled"
	CodeArchive              Code = "invalid_archive"
	CodeProcesses            Code = "process_wait"
	CodeSwap                 Code = "swap_failed"
	CodeHealth               Code = "health_failed"
	CodeRollbackPending      Code = "rollback_pending"
	CodePathRedirected       Code = "update_path_redirected"
	CodeFolderReadOnly       Code = "update_folder_read_only"
	CodeVersionInvalid       Code = "update_version_invalid"
	CodeDownloadStalled      Code = "download_stalled"
	// CodeUpdateStopped is the result of a handoff whose helper ended before it
	// changed anything (power cut, log-off, antivirus). The installed version is intact.
	CodeUpdateStopped Code = "update_interrupted"
)

// Failure retains the information needed to explain a failure without parsing
// its Message. URLs are the canonical release URL, never signed redirect URLs.
type Failure struct {
	Code          Code   `json:"code"`
	Message       string `json:"message"`
	URL           string `json:"url,omitempty"`
	Path          string `json:"path,omitempty"`
	ExpectedBytes int64  `json:"expected_bytes,omitempty"`
	ReceivedBytes int64  `json:"received_bytes,omitempty"`
	ExpectedHash  string `json:"expected_hash,omitempty"`
	ReceivedHash  string `json:"received_hash,omitempty"`
	Partial       string `json:"partial,omitempty"` // kept, removed, or absent
	Retry         string `json:"retry"`
	Via           string `json:"via,omitempty"`
	NotesURL      string `json:"notes_url,omitempty"`
}

func (f *Failure) Error() string { return fmt.Sprintf("%s: %s", f.Code, f.Message) }

func fail(code Code, message string) *Failure {
	return &Failure{Code: code, Message: message, Retry: "retry the same action"}
}

type Asset struct {
	Name          string `json:"name"`
	Kind          string `json:"kind"`
	URL           string `json:"url"`
	Size          int64  `json:"size"`
	SHA256        string `json:"sha256"`
	UnpackedBytes int64  `json:"unpacked_bytes"`
	TopFolder     string `json:"top_folder"`
}

type Notes struct {
	Summary string `json:"summary"`
	URL     string `json:"url"`
}

type Runtime struct {
	ID            string `json:"id"`
	Changed       bool   `json:"changed"`
	DownloadBytes int64  `json:"download_bytes"`
}

type Models struct {
	Changed      bool     `json:"changed"`
	NeedsPrepare []string `json:"needs_prepare"`
}

type Settings struct {
	Migrate bool `json:"migrate"`
}

type Advisory struct {
	Versions string `json:"versions"`
	Text     string `json:"text"`
}

type Manifest struct {
	SchemaVersion    int        `json:"schema_version"`
	Product          string     `json:"product"`
	Channel          string     `json:"channel"`
	Version          string     `json:"version"`
	Tag              string     `json:"tag"`
	Commit           string     `json:"commit"`
	Released         time.Time  `json:"released"`
	MinUpdatableFrom string     `json:"min_updatable_from"`
	Via              *string    `json:"via"`
	Assets           []Asset    `json:"assets"`
	Notes            Notes      `json:"notes"`
	Runtime          Runtime    `json:"runtime"`
	Models           Models     `json:"models"`
	Settings         Settings   `json:"settings"`
	Advisories       []Advisory `json:"advisories"`
	RestartRequired  bool       `json:"restart_required"`
}

type CheckStatus string

const (
	UpToDate  CheckStatus = "up_to_date"
	Available CheckStatus = "available"
	Failed    CheckStatus = "failed"
)

type Release struct {
	Version    string     `json:"version"`
	Notes      Notes      `json:"notes"`
	Size       int64      `json:"size"`
	Asset      Asset      `json:"asset"`
	Manifest   Manifest   `json:"manifest"`
	Advisories []Advisory `json:"advisories"`
}

type CheckResult struct {
	Status    CheckStatus `json:"status"`
	Available *Release    `json:"available,omitempty"`
	Failure   *Failure    `json:"failure,omitempty"`
}

type Progress struct {
	Phase         string `json:"phase"` // downloading, verifying, staging, ready
	ReceivedBytes int64  `json:"received_bytes"`
	TotalBytes    int64  `json:"total_bytes"`
}

type SpacePlan struct {
	ArchiveBytes         int64  `json:"archive_bytes"`
	ExistingPartialBytes int64  `json:"existing_partial_bytes"`
	UnpackedBytes        int64  `json:"unpacked_bytes"`
	RollbackBytes        int64  `json:"rollback_bytes"`
	AdditionalBytes      uint64 `json:"additional_bytes"`
	FreeBytes            uint64 `json:"free_bytes"`
}

type Config struct {
	InstalledVersion string
	InstallDir       string
	OldRuntimePath   string
	// Empty means the fixed GitHub defaults. Test servers supply an explicit
	// https URL, an exact host allowlist, and their httptest certificate pool.
	ManifestURL  string
	AllowedHosts []string
	RootCAs      *x509.CertPool
	// ProcessIDs is called at Install time, after the caller requested a clean
	// server stop. Include GUI and all server child PIDs; never discover by name.
	ProcessIDs    func() ([]uint32, error)
	HealthTimeout time.Duration
	WaitTimeout   time.Duration
}

// Staged is opaque: Install only accepts the package produced by this client.
type Staged struct {
	Directory      string `json:"directory"`
	Version        string `json:"version"`
	RuntimeChanged bool   `json:"runtime_changed"`
	OldRuntimePath string `json:"old_runtime_path,omitempty"`
	root           string
	asset          Asset
	seal           string
}

type Handoff struct {
	HelperPID int    `json:"helper_pid"`
	HelperExe string `json:"helper_exe"`
}

type InstallResult struct {
	Status                 string   `json:"status"` // installed, rolled_back, failed, rollback_pending
	Version                string   `json:"version,omitempty"`
	Failure                *Failure `json:"failure,omitempty"`
	RuntimeChanged         bool     `json:"runtime_changed,omitempty"`
	OldRuntimePath         string   `json:"old_runtime_path,omitempty"`
	RuntimeFollowupStarted bool     `json:"runtime_followup_started,omitempty"`
	ResultDismissed        bool     `json:"result_dismissed,omitempty"`
}
