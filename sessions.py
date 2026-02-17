from typing import Any, List, Dict, Optional

def coalesce_session_records_for_summary(
    records: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    将 session_records 做一个“折叠”视图：
    - 如果连续多条 record 的“最后一个 user 消息内容”相同，
      认为是同一轮 user 请求的多次生成（regenerate），只保留最后一次。
    - 其他情况原样保留。
    不改动原始记录，只返回一个新的 list。
    """
    result: List[Dict[str, Any]] = []
    last_user_fingerprint: Optional[str] = None

    for rec in records:
        req_msgs = rec.get("request_messages") or []
        # 找这一轮里最后一个 role == "user" 的内容
        user_texts = [
            str(m.get("content", "")) 
            for m in req_msgs 
            if m.get("role") == "user"
        ]
        last_user_text = user_texts[-1] if user_texts else None

        # 用内容本身做一个 fingerprint。以后你愿意可以换成 hash。
        fingerprint = last_user_text.strip() if last_user_text else None

        if (
            result 
            and fingerprint is not None 
            and fingerprint == last_user_fingerprint
        ):
            # 认为是同一个 user 提示的“新版尝试”，
            # 用当前这条覆盖前一条
            result[-1] = rec
        else:
            result.append(rec)

        last_user_fingerprint = fingerprint

    return result
