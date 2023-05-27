namespace py Server

// 输入
struct Request {
    1: optional list<string> Models;
    2: optional i64 currentId;
    3: optional i64 requestType;
    4: optional map<string, string> extendInfo;
}

// 输出
struct Response {
    1: optional i64 errCode=0;

    2: optional string errMsg;

    3: optional map<string, string> predictResults;
}

service PredictServer { 
    Response predict(1: required Request request);
}