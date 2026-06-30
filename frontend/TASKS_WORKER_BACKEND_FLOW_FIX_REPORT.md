# Tasks Worker Backend Flow Fix

## Fixed
- Owner task creation is now backend-first through `POST /api/tasks`.
- The frontend sends worker assignment using multiple backend-compatible fields:
  - `workerId`
  - `assignedWorker`
  - `assignedWorkerId`
  - `assignedTo`
  - `assignedToEmail`
  - `workerEmail`
- Sector assignment is sent using:
  - `sectorId`
  - `assignedSector`
  - `sector`
- Tasks are normalized from real backend IDs: `_id`, `id`, `taskId`, `task_id`, `uuid`, `mongoId`.
- Local Date.now tasks are no longer treated as backend tasks.
- The worker task page fetches `GET /api/tasks` using the logged-in worker token and shows tasks returned by the backend.
- Worker task matching now supports assignment by worker ID, email, username, name, `assignedWorker`, `assignedTo`, or `worker` object.
- Start / Done actions update backend with `PATCH /api/tasks/:id`.
- Delete button is disabled for tasks without a real backend ID, preventing invalid DELETE calls.

## Backend requirement
`GET /api/tasks` should be scoped by token:
- Owner: return tasks created by this owner.
- Worker: return tasks assigned to this worker only.

`POST /api/tasks` must return the created task with `_id` so frontend can update/delete it.
