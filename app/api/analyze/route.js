import { spawn } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

export const runtime = "nodejs";
export const maxDuration = 120;

const ROOT = process.cwd();
const UPLOAD_DIR = path.join(ROOT, "data", "uploads");

function runPython(args) {
  return new Promise((resolve, reject) => {
    const child = spawn("python", args, {
      cwd: ROOT,
      env: { ...process.env, PYTHONIOENCODING: "utf-8" },
      windowsHide: true
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString("utf8");
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString("utf8");
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(stderr || stdout || `Python exited with ${code}`));
        return;
      }
      resolve({ stdout, stderr });
    });
  });
}

export async function POST(request) {
  const form = await request.formData();
  const text = String(form.get("text") || "").trim();
  const file = form.get("file");
  const args = ["query_review_api.py", "--top-k", "100", "--min-similarity", "0.45", "--limit", "5"];

  if (file && typeof file === "object" && file.size > 0) {
    await mkdir(UPLOAD_DIR, { recursive: true });
    const ext = path.extname(file.name || "").toLowerCase();
    if (![".docx", ".pdf"].includes(ext)) {
      return Response.json({ error: "目前支持 .docx 和 .pdf 文件。" }, { status: 400 });
    }

    const safeName = `${Date.now()}-${String(file.name).replace(/[^\w.\-\u4e00-\u9fff]/g, "_")}`;
    const uploadPath = path.join(UPLOAD_DIR, safeName);
    const bytes = Buffer.from(await file.arrayBuffer());
    await writeFile(uploadPath, bytes);
    args.push("--file", uploadPath);
  } else if (text) {
    args.push("--text", text);
  } else {
    return Response.json({ error: "请上传审稿意见文件，或粘贴审稿意见文本。" }, { status: 400 });
  }

  try {
    const { stdout, stderr } = await runPython(args);
    const lines = stdout.trim().split(/\r?\n/).filter(Boolean);
    const jsonLine = [...lines].reverse().find((line) => line.trim().startsWith("{"));
    if (!jsonLine) {
      return Response.json({ error: "Python 没有返回可解析结果。", detail: stderr || stdout }, { status: 500 });
    }

    const result = JSON.parse(jsonLine);
    if (result.error) {
      return Response.json(result, { status: 422 });
    }
    return Response.json({ ...result, logs: stderr.trim() });
  } catch (error) {
    return Response.json({ error: error.message }, { status: 500 });
  }
}
