# Enrollment Modal – Data Flow (Simplified)

How the enrollment modal sends data, where it goes, and what comes back.

---

## Overview

1. **User** completes the 5-pose capture in the modal (front, left, right, up, down).
2. **Modal** sends the 5 images + user id to the **enrollment API server**.
3. **API server** forwards them to the **vectorizer/GPU backend** to get an embedding and thumbnail.
4. **API server** sends the result back to the **modal** (success, thumbnail, status).

---

## What the modal sends

**Endpoint:** `POST {apiEndpoint}/capture`  
Example: `POST /api/enrollment/capture`

**Payload (JSON):**
- `userId` – person being enrolled (or `employee_id`; API accepts either).
- `captures` – array of **5** items, one per pose. Each item has:
  - `pose` – `"front"` | `"left"` | `"right"` | `"up"` | `"down"`
  - `imageData` – base64-encoded image (e.g. JPEG from the camera).

**How:** Single HTTP POST, `Content-Type: application/json`, body = JSON above.

---

## Where it goes

- **First stop:** The **enrollment API server** (Node) that hosts routes under `/api/enrollment/`. It receives the POST and validates that there are 5 captures and a user/employee id.
- **Second stop:** The API server calls the **vectorizer backend** (e.g. GPU server). That backend does the face detection and embedding (e.g. ArcFace) and returns an embedding plus a small profile image. The backend URL is configured on the API server (e.g. `PYTHON_API_URL` or similar).

**About the vectorizer:** The vectorizer does not send the embedding or thumbnail to any other API. It only returns them in the HTTP response to the API server that called it. The **enrollment API server** is responsible for storing the result (e.g. in a transitory store), forwarding it to an IoT broker or bridge if needed, and returning success/thumbnail/status to the modal. The vectorizer is strictly request/response: no outbound callbacks or push to third-party URLs.

---

## What the modal gets back

**Response (JSON):**

- `success` – `true` if enrollment processing succeeded.
- `message` – short human-readable message (e.g. “Successfully processed 5 images…”).
- `data` (when successful):
  - `employee_id` – same id that was sent.
  - `embedding_count` – number of images used (typically 5).
  - `enrollmentPictureThumbnail` – profile/thumbnail image (path or base64, depending on setup).
  - `enrollmentStatus` – e.g. `"captured"` or `"pending"`.

**How:** Same HTTP response to the POST; the modal reads JSON from the response body. On failure, the API server returns an error status and a body with `success: false` and `error` (message).

---

## One-line summary

**Modal** → POSTs 5 base64 images + user id to **API server** → API server calls **vectorizer backend** → **API server** returns success, thumbnail, and status to the **modal**.
