package main

import (
	"reflect"
	"testing"
)

func TestParseQuota(t *testing.T) {
	good := map[string]map[string]int{
		"":                        {},
		"  ":                      {},
		"a=8":                     {"a": 8},
		"research=24, eval=16":    {"research": 24, "eval": 16},
		"a=0,b=64":                {"a": 0, "b": 64},
		"team-1=3,team_2=4":       {"team-1": 3, "team_2": 4},
		"research=24,research=30": {"research": 30},
	}
	for in, want := range good {
		got, err := parseQuota(in)
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Errorf("parseQuota(%q) = %v, %v; want %v", in, got, err, want)
		}
	}
	for _, in := range []string{"a", "a=", "=3", "a=x", "a=-1", "a=1,b", "a=1,,b=2", "a=1.5"} {
		if got, err := parseQuota(in); err == nil {
			t.Errorf("parseQuota(%q) = %v, want an error", in, got)
		}
	}
}
