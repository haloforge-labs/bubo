//go:build windows

package supervisor

import (
	"context"
	"errors"
)

// ActionResult survives Wails transport as structured data. Returning a Go
// error alone would reduce the CLI's code/remedy to a rejected string promise.
type ActionResult struct {
	Error *CLIError `json:"error"`
}

// AppError gives a GUI-owned failure a stable code while retaining its
// diagnostic cause for the collapsed Details section.
type AppError struct {
	Code   string
	Detail string
	Label  string
}

func (e *AppError) Error() string     { return e.Detail }
func (e *AppError) ErrorCode() string { return e.Code }

func appError(code, detail string) error { return &AppError{Code: code, Detail: detail} }
func appLabelError(code, detail, label string) error {
	return &AppError{Code: code, Detail: detail, Label: label}
}

func codedAction(code, detail string) ActionResult {
	return ActionResult{Error: &CLIError{Code: code, Remedy: detail}}
}

// StringResult carries an explicit user-requested value without reducing a
// failure to Wails' rejected string promise. Failed results never carry a value.
type StringResult struct {
	Value string    `json:"value"`
	Error *CLIError `json:"error"`
}

func StringResultFrom(value string, err error) StringResult {
	if err != nil {
		return StringResult{Error: ActionResultFromError(err).Error}
	}
	return StringResult{Value: value}
}

func ActionResultFromError(err error) ActionResult {
	if err == nil {
		return ActionResult{}
	}
	var command *CommandError
	if errors.As(err, &command) {
		if e, ok := command.Event.Value.(*CLIError); ok {
			v := *e
			if v.Code == "unsupported" {
				v.Code = "protocol_mismatch"
			}
			return ActionResult{Error: &v}
		}
	}
	var model *ModelError
	if errors.As(err, &model) {
		return ActionResult{Error: &CLIError{Code: model.Code, Remedy: model.Remedy, Detail: model.Detail}}
	}
	var protocol *ProtocolError
	if errors.As(err, &protocol) {
		return ActionResult{Error: &CLIError{Code: "protocol_mismatch", Remedy: protocol.Detail}}
	}
	var app *AppError
	if errors.As(err, &app) {
		return ActionResult{Error: &CLIError{Code: app.Code, Remedy: err.Error(), Label: app.Label}}
	}
	var coded interface{ ErrorCode() string }
	if errors.As(err, &coded) {
		return ActionResult{Error: &CLIError{Code: coded.ErrorCode(), Remedy: err.Error()}}
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return ActionResult{Error: &CLIError{Code: "query_timeout", Remedy: err.Error()}}
	}
	return ActionResult{Error: &CLIError{Code: "internal_error", Remedy: err.Error()}}
}

type ModelPlanResult struct {
	Plan  *ModelPlan `json:"plan"`
	Error *CLIError  `json:"error"`
}

type ModelEstimateResult struct {
	Estimate *ViewModelEstimate `json:"estimate"`
	Fit      *FitVerdict        `json:"fit,omitempty"`
	Error    *CLIError          `json:"error"`
}

func ModelEstimateResultFrom(value ModelEstimate, err error) ModelEstimateResult {
	if err != nil {
		return ModelEstimateResult{Error: ActionResultFromError(err).Error}
	}
	v := projectModelEstimate(value)
	fit := buildFitFor(value.ModelEstimateInfo)
	return ModelEstimateResult{Estimate: &v, Fit: &fit}
}
