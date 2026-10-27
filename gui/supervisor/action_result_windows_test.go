//go:build windows

package supervisor

import (
	"context"
	"fmt"
	"testing"
)

func TestActionResultPreservesWrappedQueryTimeout(t *testing.T) {
	err := fmt.Errorf("status query: %w", context.DeadlineExceeded)
	result := ActionResultFromError(err)
	if result.Error == nil || result.Error.Code != "query_timeout" || result.Error.Remedy != err.Error() {
		t.Fatal("deadline lost its typed code/remedy", result)
	}
}

func TestStringResultFailureNeverContainsSuccessValue(t *testing.T) {
	result := StringResultFrom("sensitive partial value", context.DeadlineExceeded)
	if result.Value != "" || result.Error == nil || result.Error.Code != "query_timeout" {
		t.Fatal("failed string result leaked a value or lost typed error", result.Error)
	}
	if result := StringResultFrom("requested value", nil); result.Value != "requested value" || result.Error != nil {
		t.Fatal("successful string result changed value", result)
	}
}
