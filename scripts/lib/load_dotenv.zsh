#!/bin/zsh

# Load simple KEY=VALUE dotenv files without evaluating their contents as shell code.
load_dotenv_file() {
  emulate -L zsh
  setopt extended_glob
  local env_file="$1"
  local line key value
  [[ -f "$env_file" ]] || return 0

  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line##[[:space:]]#}"
    line="${line%%[[:space:]]#}"
    [[ -z "$line" || "$line" == \#* || "$line" != *\=* ]] && continue
    key="${line%%=*}"
    value="${line#*=}"
    key="${key##[[:space:]]#}"
    key="${key%%[[:space:]]#}"
    value="${value##[[:space:]]#}"
    value="${value%%[[:space:]]#}"
    [[ "$key" == [A-Za-z_][A-Za-z0-9_]# ]] || continue
    (( ${+parameters[$key]} )) && continue
    if (( ${#value} >= 2 )) && {
      [[ "$value[1]" == '"' && "$value[-1]" == '"' ]] ||
      [[ "$value[1]" == "'" && "$value[-1]" == "'" ]]
    }; then
      value="${value[2,-2]}"
    fi
    typeset -gx "$key=$value"
  done < "$env_file"
}
