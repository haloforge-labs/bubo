// viewgen emits explicit copies and typed projections. Go has no standard
// recursive struct copier; generating assignments keeps ownership reviewable
// and avoids reflection and JSON round trips on every poll.
package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"os"
	"sort"
	"strings"
)

var structs = map[string]*ast.StructType{}
var types = map[string]ast.Expr{}
var fset = token.NewFileSet()
var copies, projections = map[string]bool{}, map[string]bool{}
var body bytes.Buffer
var serial int

func syntax(n ast.Node) string        { var b bytes.Buffer; _ = format.Node(&b, fset, n); return b.String() }
func raw(t ast.Expr) bool             { return syntax(t) == "json.RawMessage" }
func nullableStrings(t ast.Expr) bool { return syntax(t) == "map[string]*string" }
func needs(t ast.Expr, seen map[string]bool) bool {
	if raw(t) {
		return true
	}
	switch x := t.(type) {
	case *ast.Ident:
		if seen[x.Name] {
			return false
		}
		seen[x.Name] = true
		if s := structs[x.Name]; s != nil {
			return needs(s, seen)
		}
	case *ast.StarExpr:
		return needs(x.X, seen)
	case *ast.ArrayType:
		return needs(x.Elt, seen)
	case *ast.MapType:
		return needs(x.Value, seen)
	case *ast.StructType:
		for _, f := range x.Fields.List {
			if needs(f.Type, seen) {
				return true
			}
		}
	}
	return false
}
func typename(t ast.Expr, project bool) string {
	if project && nullableStrings(t) {
		return "map[string]Value"
	}
	if project && raw(t) {
		return "Value"
	}
	switch x := t.(type) {
	case *ast.Ident:
		if project && needs(x, map[string]bool{}) {
			return "View" + x.Name
		}
	case *ast.StarExpr:
		return "*" + typename(x.X, project)
	case *ast.ArrayType:
		return "[]" + typename(x.Elt, project)
	case *ast.MapType:
		return "map[" + syntax(x.Key) + "]" + typename(x.Value, project)
	case *ast.StructType:
		var b strings.Builder
		b.WriteString("struct {\n")
		for _, f := range x.Fields.List {
			if len(f.Names) == 0 {
				fmt.Fprintf(&b, "%s %s\n", typename(f.Type, project), tag(f))
			}
			for _, n := range f.Names {
				if ast.IsExported(n.Name) {
					fmt.Fprintf(&b, "%s %s %s\n", n.Name, typename(f.Type, project), tag(f))
				}
			}
		}
		b.WriteString("}")
		return b.String()
	}
	return syntax(t)
}
func tag(f *ast.Field) string {
	if f.Tag != nil {
		return f.Tag.Value
	}
	return ""
}
func emit(t ast.Expr, dst, src string, project bool) {
	if project && nullableStrings(t) {
		fmt.Fprintf(&body, "if %s != nil { %s = make(map[string]Value,len(%s)); for k,v := range %s { %s[k] = valueFromNullableString(v) }}\n", src, dst, src, src, dst)
		return
	}
	if project && raw(t) {
		fmt.Fprintf(&body, "%s = valueFromJSON(%s)\n", dst, src)
		return
	}
	switch x := t.(type) {
	case *ast.Ident:
		if structs[x.Name] != nil {
			p := project && needs(x, map[string]bool{})
			name := "copy" + x.Name
			if p {
				projections[x.Name] = true
				name = "project" + x.Name
			} else {
				copies[x.Name] = true
			}
			fmt.Fprintf(&body, "%s = %s(%s)\n", dst, name, src)
			return
		}
	case *ast.StarExpr:
		fmt.Fprintf(&body, "if %s != nil { %s = new(%s)\n", src, dst, typename(x.X, project))
		emit(x.X, "(*"+dst+")", "(*"+src+")", project)
		body.WriteString("}\n")
		return
	case *ast.ArrayType:
		serial++
		i := fmt.Sprintf("i%d", serial)
		fmt.Fprintf(&body, "if %s != nil { %s = make(%s,len(%s)); for %s := range %s {\n", src, dst, typename(t, project), src, i, src)
		emit(x.Elt, dst+"["+i+"]", src+"["+i+"]", project)
		body.WriteString("}}\n")
		return
	case *ast.MapType:
		serial++
		k, v := fmt.Sprintf("k%d", serial), fmt.Sprintf("v%d", serial)
		fmt.Fprintf(&body, "if %s != nil { %s = make(%s,len(%s)); for %s,%s := range %s { var item %s\n", src, dst, typename(t, project), src, k, v, src, typename(x.Value, project))
		emit(x.Value, "item", v, project)
		fmt.Fprintf(&body, "%s[%s] = item }}\n", dst, k)
		return
	case *ast.StructType:
		for _, f := range x.Fields.List {
			if len(f.Names) == 0 {
				n := syntax(f.Type)
				d := n
				if project {
					d = typename(f.Type, true)
				}
				emit(f.Type, dst+"."+d, src+"."+n, project)
			}
			for _, n := range f.Names {
				if !project || ast.IsExported(n.Name) {
					emit(f.Type, dst+"."+n.Name, src+"."+n.Name, project)
				}
			}
		}
		return
	}
	// RawMessage is the only mutable selector type in these contract structs.
	if raw(t) {
		fmt.Fprintf(&body, "%s = append(json.RawMessage(nil), %s...)\n", dst, src)
	} else {
		fmt.Fprintf(&body, "%s = %s\n", dst, src)
	}
}
func main() {
	files, err := parser.ParseDir(fset, ".", func(f os.FileInfo) bool {
		return strings.HasSuffix(f.Name(), ".go") && !strings.HasSuffix(f.Name(), "_test.go") && !strings.HasSuffix(f.Name(), "_generated_windows.go")
	}, 0)
	if err != nil {
		panic(err)
	}
	for _, f := range files["supervisor"].Files {
		for _, d := range f.Decls {
			if g, ok := d.(*ast.GenDecl); ok {
				for _, s := range g.Specs {
					if t, ok := s.(*ast.TypeSpec); ok {
						types[t.Name.Name] = t.Type
						if st, ok := t.Type.(*ast.StructType); ok {
							structs[t.Name.Name] = st
						}
					}
				}
			}
		}
	}
	var names []string
	for name := range structs {
		names = append(names, name)
	}
	for _, name := range names {
		if !strings.HasPrefix(name, "View") && needs(ast.NewIdent(name), map[string]bool{}) {
			e, err := parser.ParseExpr(typename(structs[name], true))
			if err != nil {
				panic(err)
			}
			structs["View"+name] = e.(*ast.StructType)
		}
	}
	copies["Snapshot"] = true
	copies["View"] = true
	copies["ModelsState"] = true
	for _, name := range []string{"Instance", "Setting", "Switch", "EffectiveSwitch", "MemoryEstimate", "InferenceDefaults", "ThinkMarkerBias", "LoopBreaker", "ModelsState", "ModelEstimateInfo", "ModelEstimate"} {
		if needs(ast.NewIdent(name), map[string]bool{}) {
			projections[name] = true
		}
	}
	done := map[string]bool{}
	for {
		var pending []string
		for n := range copies {
			if !done["copy"+n] {
				pending = append(pending, "copy"+n)
			}
		}
		for n := range projections {
			if !done["project"+n] {
				pending = append(pending, "project"+n)
			}
		}
		sort.Strings(pending)
		if len(pending) == 0 {
			break
		}
		for _, fn := range pending {
			done[fn] = true
			project := strings.HasPrefix(fn, "project")
			name := strings.TrimPrefix(strings.TrimPrefix(fn, "project"), "copy")
			out := name
			if project {
				out = "View" + name
				fmt.Fprintf(&body, "type %s %s\n", out, typename(structs[name], true))
			}
			fmt.Fprintf(&body, "func %s(in %s) (out %s) {\n", fn, name, out)
			emit(structs[name], "out", "in", project)
			body.WriteString("return\n}\n")
		}
	}
	header := "// Code generated by go run ./internal/viewgen; DO NOT EDIT.\n//go:build windows\n\npackage supervisor\n\nimport \"encoding/json\"\n\n"
	result, err := format.Source(append([]byte(header), body.Bytes()...))
	if err != nil {
		os.WriteFile("view_data_generated_windows.go", append([]byte(header), body.Bytes()...), 0600)
		panic(err)
	}
	if err = os.WriteFile("view_data_generated_windows.go", result, 0644); err != nil {
		panic(err)
	}
}
