"""接口冒烟测试：不需要已建好的模型也能跑。

用法：
    python tools/smoke_test.py                # 默认 http://127.0.0.1:8000
    python tools/smoke_test.py --base http://127.0.0.1:8100

覆盖：健康检查、任务列表、非法任务、首页与静态资源、阈值读写、导出、清空。
未加载模型时自动跳过推理相关用例。
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

PASS = 0
FAIL = 0


def req(base, path, method="GET", data=None, headers=None, timeout=15):
    url = base.rstrip("/") + path
    body = None
    hdrs = dict(headers or {})
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=body, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, raw
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"[PASS] {name}" + (f" | {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"[FAIL] {name}" + (f" | {detail}" if detail else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    base = args.base

    # 1 健康检查
    code, raw = req(base, "/health")
    ok = code == 200 and json.loads(raw).get("status") == "ok"
    check("健康检查", ok, f"status={code}")
    if not ok:
        print("\n服务没起来，先运行 start_web.bat / start_web.sh")
        return 1

    # 2 任务列表
    code, raw = req(base, "/api/tasks")
    data = json.loads(raw) if code == 200 else {}
    tasks = data.get("tasks", [])
    check("任务列表", code == 200 and len(tasks) > 0, f"{len(tasks)} 个任务")
    if not tasks:
        return 1

    task = tasks[0]["task"]

    # 3 任务详情字段
    keys = {"task", "display_name", "has_model", "threshold"}
    check("任务详情含必要字段", keys <= set(tasks[0]), f"{sorted(keys - set(tasks[0])) or '齐全'}")

    # 4 非法任务返回 404
    code, _ = req(base, "/api/tasks/__nope__/state")
    check("非法任务返回 404", code == 404, f"status={code}")

    # 5 首页
    code, raw = req(base, "/")
    check("首页可访问", code == 200 and b"<html" in raw.lower(), f"status={code}")

    # 6 静态资源
    code, _ = req(base, "/static/app.js")
    check("静态资源可访问", code == 200, f"status={code}")

    # 7 阈值读取
    code, raw = req(base, f"/api/tasks/{task}/state")
    st = json.loads(raw) if code == 200 else {}
    thr = st.get("threshold") or {}
    check("状态接口返回阈值", code == 200 and "image_score" in thr,
          f"{thr.get('image_score')}/{thr.get('pixel_score')}")

    # 8 阈值写入后能读回
    if thr:
        old = {"image_score": thr["image_score"], "pixel_score": thr["pixel_score"]}
        probe = {"image_score": round(old["image_score"] + 0.01, 6),
                 "pixel_score": round(old["pixel_score"] + 0.01, 6)}
        code, _ = req(base, f"/api/tasks/{task}/threshold", "POST", probe)
        code2, raw2 = req(base, f"/api/tasks/{task}/state")
        got = (json.loads(raw2).get("threshold") or {}).get("image_score")
        check("阈值可写回", code == 200 and abs(got - probe["image_score"]) < 1e-6,
              f"写入 {probe['image_score']} 读回 {got}")
        req(base, f"/api/tasks/{task}/threshold", "POST", old)

    # 9 CSV 导出
    code, raw = req(base, f"/api/tasks/{task}/export?fmt=csv")
    check("CSV 导出", code == 200 and raw.startswith(b"\xef\xbb\xbf"), f"{len(raw)} 字节")

    # 10 JSON 导出
    code, raw = req(base, f"/api/tasks/{task}/export?fmt=json")
    ok_json = False
    try:
        ok_json = isinstance(json.loads(raw), (list, dict))
    except Exception:
        pass
    check("JSON 导出", code == 200 and ok_json, f"{len(raw)} 字节")

    # 11 模型信息
    code, raw = req(base, f"/api/tasks/{task}/model")
    minfo = json.loads(raw) if code == 200 else {}
    has_model = bool(minfo.get("has_model"))
    check("模型信息接口", code == 200, "已加载" if has_model else "未建模型")

    # 12 未建模型时推理应返回 503
    if not has_model:
        code, _ = req(base, f"/api/infer/{task}/frame", "POST", {"image": "data:image/png;base64,iVBORw0KGgo="})
        check("无模型时推理返回 503", code == 503, f"status={code}")

    # 13 清空历史
    code, _ = req(base, f"/api/tasks/{task}/clear", "POST", {})
    code2, raw2 = req(base, f"/api/tasks/{task}/state")
    total = (json.loads(raw2).get("stats") or {}).get("total", -1)
    check("清空历史", code == 200 and total == 0, f"剩 {total} 条")

    print(f"\n合计 {PASS + FAIL} 项，通过 {PASS}，失败 {FAIL}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
