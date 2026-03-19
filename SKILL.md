# SKILL: Gateway — RAG + Multi-Model + Multi-Project

## Base URL
```
http://localhost:8080
```

## Flow chuẩn
```
1. POST /task           → RAG + model → nhận JSON diff
2. Apply diff           → write vào filesystem
3. POST /confirm-apply  → re-index file (KHÔNG bỏ bước này)
```

## POST /task
```json
{
  "task": "mô tả rõ ràng việc cần làm",
  "project": "tên-project",
  "model": "gpt-4.1",
  "top_k": 5
}
```

Response:
```json
{
  "diff": {
    "file": "src/services/userService.ts",
    "changes": [
      { "type": "replace", "start_line": 42, "end_line": 47, "new_content": "..." }
    ]
  },
  "model_used": "gpt-4.1",
  "project": "my-app",
  "rag_sources": ["src/services/userService.ts"]
}
```

## POST /confirm-apply
```json
{ "project": "my-app", "file_path": "src/services/userService.ts" }
```

## Utility
```
GET  /projects          → list projects
GET  /models            → list models + providers
POST /index/full        → index toàn bộ project (lần đầu setup)
POST /index/file        → re-index 1 file thủ công
GET  /health            → health + bootstrap status
```

## Models available
| Model | Provider |
|---|---|
| gpt-4.1, gpt-4.1-mini | OpenAI |
| gemini-2.5-pro, gemini-2.5-flash | Google |
| copilot-gpt-4.1, copilot-claude | GitHub Copilot |

## Error handling
| Status | Hành động |
|---|---|
| 404 project | Check GET /projects |
| 404 no code | Chạy POST /index/full trước |
| 409 diff fail | Gọi POST /index/file rồi retry |
| 500 | Retry 1 lần, nếu fail thì report |
