// The real bridge returns JSON, without the converter methods on generated Go
// classes. Test replies intentionally exercise that wire shape at the boundary.
export function bridgeReply<Result>(json: unknown): Result {
  return json as Result
}
