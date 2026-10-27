module bubo/gui/supervisor

go 1.27.2

require (
	bubo/gui/updater v0.0.0
	github.com/danieljoos/wincred v1.2.3
	github.com/santhosh-tekuri/jsonschema/v6 v6.0.3
	golang.org/x/sys v0.46.0
)

require golang.org/x/text v0.14.0 // indirect

replace bubo/gui/updater => ../updater
