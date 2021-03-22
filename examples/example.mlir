module @testmodule  {
  func @main() {
    %11 = db.constant ("0.01") :!db.float<32>
    //%12 = db.constant ("0.01") :!db.float<32>
    %13 = db.sub %11 : !db.float<32>,%11 : !db.float<32>
    return
  }
}