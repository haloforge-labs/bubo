//go:build windows

package supervisor

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"sync"

	"github.com/danieljoos/wincred"
)

// KeyTarget is the Windows Credential Manager target name of the serve API key.
const KeyTarget = "Bubo/serve-api-key"

var keyTarget = KeyTarget

// KeyMask is the only text a window, log or snapshot may show for the key.
const KeyMask = "••••"

// LANUnavailable is the message when the key cannot be stored. LAN access then
// stays off. The GUI never keeps the key anywhere else.
const LANUnavailable = "LAN access unavailable: cannot store the key"

var (
	// ErrCredentialNotFound means the store holds no entry for the target.
	ErrCredentialNotFound = errors.New("credential not found")
	// ErrNoKey means no API key exists yet (LAN access was never turned on).
	ErrNoKey = appError("key_not_created", "No API key exists. Turn on local network access first.")
)

// CredentialStore is the secret storage behind the API key. Production uses
// Windows Credential Manager; tests use a fake.
type CredentialStore interface {
	Read(target string) (string, error)
	Write(target, secret string) error
	Delete(target string) error
}

// WindowsCredentialStore keeps generic credentials in this user's vault.
type WindowsCredentialStore struct{}

func keyCredential(target, secret string) (*wincred.GenericCredential, error) {
	blob := []byte(secret)
	if len(blob) == 0 || len(blob) > 2560 {
		return nil, appError("key_store_failed", "credential size is not valid")
	}
	c := wincred.NewGenericCredential(target)
	c.Persist = wincred.PersistLocalMachine
	c.UserName = "Bubo"
	c.CredentialBlob = blob
	return c, nil
}

func (WindowsCredentialStore) Write(target, secret string) error {
	c, err := keyCredential(target, secret)
	if err != nil {
		return err
	}
	if err := c.Write(); err != nil {
		return fmt.Errorf("%w: CredWrite: %v", appError("key_store_failed", "Credential write failed"), err)
	}
	return nil
}

func (WindowsCredentialStore) Read(target string) (string, error) {
	c, err := wincred.GetGenericCredential(target)
	if errors.Is(err, wincred.ErrElementNotFound) {
		return "", ErrCredentialNotFound
	}
	if err != nil {
		return "", fmt.Errorf("%w: CredRead: %v", appError("key_store_failed", "Credential read failed"), err)
	}
	if len(c.CredentialBlob) == 0 {
		return "", ErrCredentialNotFound
	}
	return string(c.CredentialBlob), nil
}

func (WindowsCredentialStore) Delete(target string) error {
	err := wincred.NewGenericCredential(target).Delete()
	if errors.Is(err, wincred.ErrElementNotFound) {
		return ErrCredentialNotFound
	}
	if err != nil {
		return fmt.Errorf("%w: CredDelete: %v", appError("key_store_failed", "Credential delete failed"), err)
	}
	return nil
}

// KeyManager owns the serve API key. The key is 32 random bytes in hex (64
// characters: no leading dash, safe on a command line). It lives only in the
// credential store; this type returns it only to Ensure (for the launch) and
// Reveal (for an explicit user action).
type KeyManager struct {
	mu      sync.Mutex
	store   CredentialStore
	target  string
	present int8 // -1 unknown, 0 no, 1 yes
}

// NewKeyManager uses the production target name.
func NewKeyManager(store CredentialStore) *KeyManager {
	return NewKeyManagerFor(store, keyTarget)
}

// NewKeyManagerFor lets a test use a throwaway target name.
func NewKeyManagerFor(store CredentialStore, target string) *KeyManager {
	if store == nil {
		store = WindowsCredentialStore{}
	}
	return &KeyManager{store: store, target: target, present: -1}
}

func newKey() (string, error) {
	var b [32]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(b[:]), nil
}

// Present says whether a key is stored. It never returns the key.
func (k *KeyManager) Present() bool {
	k.mu.Lock()
	defer k.mu.Unlock()
	if k.present < 0 {
		if v, err := k.store.Read(k.target); err == nil && v != "" {
			k.present = 1
		} else {
			k.present = 0
		}
	}
	return k.present == 1
}

func (k *KeyManager) write() (string, error) {
	key, err := newKey()
	if err != nil {
		return "", appError("key_store_failed", LANUnavailable)
	}
	if err = k.store.Write(k.target, key); err != nil {
		return "", errors.New(LANUnavailable)
	}
	// A store that accepts a write but cannot read it back is not usable.
	if back, err := k.store.Read(k.target); err != nil || back != key {
		return "", errors.New(LANUnavailable)
	}
	k.present = 1
	return key, nil
}

// Ensure returns the stored key, or creates and stores one. A failure of the
// store returns LANUnavailable; there is no plaintext fallback.
func (k *KeyManager) Ensure() (string, error) {
	k.mu.Lock()
	defer k.mu.Unlock()
	v, err := k.store.Read(k.target)
	if err == nil && v != "" {
		k.present = 1
		return v, nil
	}
	if err != nil && !errors.Is(err, ErrCredentialNotFound) {
		return "", errors.New(LANUnavailable)
	}
	return k.write()
}

// Reveal returns the stored key. Call it only for an explicit user action.
func (k *KeyManager) Reveal() (string, error) {
	k.mu.Lock()
	defer k.mu.Unlock()
	v, err := k.store.Read(k.target)
	if errors.Is(err, ErrCredentialNotFound) || err == nil && v == "" {
		k.present = 0
		return "", ErrNoKey
	}
	if err != nil {
		return "", appError("key_store_failed", "The key store could not be read.")
	}
	k.present = 1
	return v, nil
}

// Reset replaces the stored key with a new one. The old key stops working at
// the next server start. It creates a key when none exists.
func (k *KeyManager) Reset() error {
	k.mu.Lock()
	defer k.mu.Unlock()
	_, err := k.write()
	return err
}

// Delete removes the stored key. A missing key is not an error.
func (k *KeyManager) Delete() error {
	k.mu.Lock()
	defer k.mu.Unlock()
	err := k.store.Delete(k.target)
	if err != nil && !errors.Is(err, ErrCredentialNotFound) {
		return appError("key_store_failed", "The key store could not be changed.")
	}
	k.present = 0
	return nil
}

// scrubKey replaces every copy of the key in b with KeyMask.
func scrubKey(b []byte, key string) []byte {
	if key == "" {
		return b
	}
	return bytes.ReplaceAll(b, []byte(key), []byte(KeyMask))
}

// Scrub replaces the stored key in b with KeyMask. It reads the store only when
// a key exists, and it keeps the key in memory for no longer than the call.
func (k *KeyManager) Scrub(b []byte) []byte {
	if k == nil || len(b) == 0 || !k.Present() {
		return b
	}
	k.mu.Lock()
	v, err := k.store.Read(k.target)
	k.mu.Unlock()
	if err != nil {
		return b
	}
	return scrubKey(b, v)
}

// ScrubString is Scrub for text.
func (k *KeyManager) ScrubString(s string) string {
	if s == "" {
		return s
	}
	return string(k.Scrub([]byte(s)))
}
