module bubo/gui/tests

go 1.27.2

require (
	bubo/gui/supervisor v0.0.0
	golang.org/x/sys v0.46.0
)

require (
	bubo/gui/updater v0.0.0 // indirect
	github.com/danieljoos/wincred v1.2.3 // indirect
	github.com/santhosh-tekuri/jsonschema/v6 v6.0.3 // indirect
	golang.org/x/text v0.14.0 // indirect
)

replace bubo/gui/supervisor => ../supervisor

replace bubo/gui/updater => ../updater
