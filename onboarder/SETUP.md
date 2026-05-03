# Onboarder setup

One-time manual setup needed before the wizard mode works.
After this is done, you only ever touch the Google Sheet (to edit `Criteria` and approve rows in `Run-*` tabs).

---

## 1. Create the Google Sheet

1. Open https://sheets.google.com → **Blank**.
2. Name it: `NewMindStart Onboarder`.
3. Copy the **Sheet ID** from the URL — the long random string between `/d/` and `/edit`:
   ```
   https://docs.google.com/spreadsheets/d/SHEET_ID_HERE/edit
   ```
   You'll need this in the gateway config.

### 1a. Tab `Criteria`

Rename the default `Sheet1` to **`Criteria`**. Fill in two columns:

| A: Параметр | B: Значение |
|---|---|
| min_subscribers | 5000 |
| max_subscribers | 500000 |
| min_videos_on_channel | 20 |
| max_videos_on_channel | 1000 |
| max_video_age_months | 24 |
| preferred_languages | ru, en, es |
| preferred_video_length_min | 10 |
| preferred_video_length_max | 35 |

Edit any time — the bot reads this on each `/menu → 🎓 Новый курс` run.

### 1b. Tab `Runs`

Add a new tab named **`Runs`**. First row (header):

| A: timestamp | B: topic | C: count | D: status | E: sheet_tab | F: courses |
|---|---|---|---|---|---|

The bot writes rows automatically.

### 1c. Run-* tabs

These are auto-created by the bot. You don't pre-create them.

---

## 2. Create the Google service account

The bot authenticates to Google Sheets via a **service account** (no OAuth pop-ups).

1. Open https://console.cloud.google.com/
2. Top bar: project dropdown → **New Project** → name it `newmindstart-onboarder` → **Create**.
3. Wait ~30 sec, then switch into that project.
4. Left sidebar: **APIs & Services → Library**:
   - Search **Google Sheets API** → click → **Enable**
   - Search **Google Drive API** → click → **Enable**  
     *(gspread needs Drive scope to open spreadsheets by ID)*
5. Left sidebar: **APIs & Services → Credentials → Create Credentials → Service account**.
   - Service account name: `onboarder-bot`
   - Click **Create and Continue**
   - Role: **Editor** (or skip — not needed for Sheets-only access)
   - Click **Done**
6. Click on the new service account in the list → tab **Keys** → **Add Key → Create new key**:
   - Key type: **JSON**
   - **Create** → file downloads automatically
7. Note the service account **email** — looks like `onboarder-bot@newmindstart-onboarder.iam.gserviceaccount.com`. You'll need it in the next step.

---

## 3. Share the Sheet with the service account

Service accounts can't access your Sheet by default. Share it explicitly:

1. Go back to your Google Sheet.
2. Click **Share** (top right).
3. Paste the service account email (`onboarder-bot@...iam.gserviceaccount.com`).
4. Set role: **Editor**.
5. **Untick** "Notify people" (it's a bot, no email needed).
6. **Share**.

---

## 4. Put credentials on the VPS

```bash
ssh jarvis-vps

# Move the JSON key from your laptop to VPS
# (run from your laptop, not from VPS):
scp ~/Downloads/newmindstart-onboarder-*.json jarvis-vps:~/.secrets/google-sheets-sa.json

# On VPS — lock down permissions:
chmod 600 ~/.secrets/google-sheets-sa.json
```

---

## 5. Wire up `config.json` on VPS

Edit `~/projects/jarvis-telegram-gateway/config.json`. Inside the agent that should have wizard mode (e.g. `silvana`), add:

```json
"onboarder": {
  "enabled": true,
  "google_sheet_id": "PASTE_SHEET_ID_FROM_STEP_1_HERE",
  "google_service_account_file": "~/.secrets/google-sheets-sa.json",
  "nms_endpoint": "https://truelifeflow.com",
  "nms_api_token_file": "~/.secrets/nms-agent-token",
  "openai_api_key_file": "~/.secrets/openai.key",
  "elevenlabs_api_key_file": "~/.secrets/elevenlabs.key",
  "bunny_stream_library_id": "YOUR_BUNNY_LIBRARY_ID",
  "bunny_stream_api_key_file": "~/.secrets/bunny-stream.key",
  "scratch_dir": "/tmp/onboarder",
  "phase2_parallel_videos": 1
}
```

The remaining secret files (NMS token, OpenAI, ElevenLabs, Bunny) are added later as Phase 1/2 are implemented — see deploy notes when each is wired in.

---

## 6. Restart gateway

```bash
ssh jarvis-vps
sudo systemctl restart jarvis-gateway
sudo journalctl -u jarvis-gateway -f
```

In Telegram: `/menu` → you should now see the **🎓 Новый курс** button.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `🎓 Новый курс` button missing | `onboarder.enabled` not set to `true`, or config not reloaded → restart gateway |
| `gspread.exceptions.SpreadsheetNotFound` | Sheet ID wrong, or not shared with service account email |
| `google.auth.exceptions.RefreshError` | JSON key file path wrong, or file permissions block jarvis user from reading |
| `403 PERMISSION_DENIED on Sheets API` | Sheets API or Drive API not enabled in the Cloud project |
