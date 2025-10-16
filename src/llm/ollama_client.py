import os
import time
import json
from typing import Dict, List, Optional, Tuple, Any

import requests

class OllamaClient:
	def __init__(self) -> None:
		# Use env; default to the remote you curl against to avoid localhost hangs
		self.base = os.getenv("OLLAMA_BASE_URL", "http://popp-llm01.server.lan:11434").rstrip("/")
		self.model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b-instruct-q4_K_M")
		# Split timeouts: fast connect, bounded read
		self.connect_timeout = int(os.getenv("OLLAMA_CONNECT_TIMEOUT", "5"))
		self.read_timeout = int(os.getenv("OLLAMA_TIMEOUT", "60"))  # CHANGED default 60s
		self.total_timeout = int(os.getenv("OLLAMA_TOTAL_TIMEOUT", "60"))  # NEW
		self.stream_mode = os.getenv("OLLAMA_STREAM_MODE", "0").lower() not in ("0", "false", "no")  # NEW
		self.stream_read_timeout = int(os.getenv("OLLAMA_STREAM_READ_TIMEOUT", "15"))  # NEW
		self.retries = max(0, int(os.getenv("OLLAMA_RETRIES", "1")))
		self._sess = requests.Session()

	def _timeout(self) -> Tuple[int, int]:
		return (self.connect_timeout, self.read_timeout)

	def _preflight(self, verbose: bool) -> bool:
		"""
		Fast health check to avoid long hangs when host is unreachable.
		"""
		try:
			url = f"{self.base}/api/tags"
			r = self._sess.get(url, timeout=(self.connect_timeout, 5))
			r.raise_for_status()
			return True
		except Exception as e:
			if verbose:
				print(f"[ollama] preflight failed: {e} (base={self.base})")
			return False

	def list_models(self, verbose: bool = False) -> List[str]:
		try:
			r = requests.get(self.base + "/api/tags", timeout=10)
			r.raise_for_status()
			data = r.json()
			return [m.get("name") or m.get("model") for m in data.get("models") or []]
		except Exception as e:
			if verbose:
				print(f"[llm] list_models failed: {e}")
			return []

	def _stream_post(self, url: str, payload: Dict, verbose: bool = False) -> Tuple[str, bool]:
		buf: List[str] = []
		done = False
		with requests.post(url, json=payload, timeout=self._timeout(), stream=True) as r:
			if r.status_code != 200:
				if verbose:
					print(f"[llm] http {r.status_code}: {r.text[:300]}")
				r.raise_for_status()
			for line in r.iter_lines(decode_unicode=True):
				if not line:
					continue
				try:
					obj = json.loads(line)
				except Exception:
					try:
						obj = r.json()
					except Exception:
						continue
				if "message" in obj and isinstance(obj["message"], dict):
					part = (obj["message"].get("content") or "")
					if part:
						buf.append(part)
				if "response" in obj:
					part = (obj.get("response") or "")
					if part:
						buf.append(part)
				if obj.get("done") is True:
					done = True
		return "".join(buf), done

	def _chat(self, system: str, user: str, temperature: float, verbose: bool) -> Optional[str]:
		url = f"{self.base}/api/chat"
		payload: Dict[str, Any] = {
			"model": self.model,
			"messages": [
				{"role": "system", "content": system},
				{"role": "user", "content": user},
			],
			"stream": False,
			"options": {"temperature": temperature},
		}
		if verbose:
			print(f"[ollama] POST {url} (chat) model={self.model}")
		r = self._sess.post(url, json=payload, timeout=self._timeout())
		r.raise_for_status()
		data = r.json() or {}
		msg = data.get("message") or {}
		content = (msg.get("content") or "").strip()
		return content or None

	def _generate(self, system: str, user: str, temperature: float, verbose: bool) -> Optional[str]:
		url = f"{self.base}/api/generate"
		prompt = f"{system.strip()}\n\n{user.strip()}\n"
		payload: Dict[str, Any] = {
			"model": self.model,
			"prompt": prompt,
			"stream": False,
			"options": {"temperature": temperature},
		}
		if verbose:
			print(f"[ollama] POST {url} (generate) model={self.model}")
		r = self._sess.post(url, json=payload, timeout=self._timeout())
		r.raise_for_status()
		data = r.json() or {}
		content = (data.get("response") or "").strip()
		return content or None

	def _generate_stream(self, system: str, user: str, temperature: float, verbose: bool) -> Optional[str]:  # NEW
		"""
		Streaming fallback for /api/generate with an overall deadline and per-chunk read timeout.
		"""
		url = f"{self.base}/api/generate"
		prompt = f"{system.strip()}\n\n{user.strip()}\n"
		payload: Dict[str, Any] = {
			"model": self.model,
			"prompt": prompt,
			"stream": True,
			"options": {"temperature": temperature},
		}
		if verbose:
			print(f"[ollama] POST {url} (generate, stream) model={self.model}")

		start = time.monotonic()
		buf: List[str] = []
		last_activity = start
		with self._sess.post(url, json=payload, timeout=(self.connect_timeout, self.stream_read_timeout), stream=True) as r:
			r.raise_for_status()
			for line in r.iter_lines(decode_unicode=True):
				now = time.monotonic()
				if now - start > self.total_timeout:
					if verbose:
						print("[ollama] streaming deadline reached")
					break
				if not line:
					# inactivity guard
					if now - last_activity > self.stream_read_timeout:
						if verbose:
							print("[ollama] streaming inactivity timeout")
						break
					continue
				last_activity = now
				try:
					obj = json.loads(line)
				except Exception:
					# ignore malformed line
					continue
				if "response" in obj:
					part = (obj.get("response") or "")
					if part:
						buf.append(part)
				# Some servers send "message.content" even on generate
				if "message" in obj and isinstance(obj["message"], dict):
					part = (obj["message"].get("content") or "")
					if part:
						buf.append(part)
				if obj.get("done") is True:
					break
		text = "".join(buf).strip()
		return text or None

	def complete(self, system: str, user: str, prefer: Optional[str] = None, temperature: float = 0.2, verbose: bool = False) -> Optional[str]:
		if verbose:
			print(f"[ollama] base={self.base} prefer={prefer} model={self.model} retries={self.retries}")
		# quick preflight to fail fast if unreachable
		if not self._preflight(verbose):
			return None

		def attempt(call_fn):
			delay = 0.8
			deadline = time.monotonic() + self.total_timeout  # overall attempt budget
			for i in range(max(1, self.retries + 1)):
				try:
					# stop if we exceeded total budget
					if time.monotonic() > deadline:
						if verbose:
							print("[ollama] total deadline reached before attempt")
						return None
					return call_fn()
				except requests.exceptions.ReadTimeout as e:
					if verbose:
						print(f"[ollama] attempt {i+1} read-timeout: {e}")
					time.sleep(delay)
					delay *= 1.5
				except Exception as e:
					if verbose:
						print(f"[ollama] attempt {i+1} failed: {e}")
					time.sleep(delay)
					delay *= 1.5
			return None

		# Helper that tries non-stream first, then stream
		def call_generate_combo():
			if self.stream_mode:
				# user forces streaming
				return self._generate_stream(system, user, temperature, verbose)
			out = None
			try:
				out = self._generate(system, user, temperature, verbose)
			except requests.exceptions.ReadTimeout:
				out = None
			except Exception:
				out = None
			if out:
				return out
			# fallback to streaming
			if verbose:
				print("[ollama] falling back to streaming generate")
			return self._generate_stream(system, user, temperature, verbose)

		m = (prefer or "auto").lower()
		if m == "chat":
			return attempt(lambda: self._chat(system, user, temperature, verbose))
		if m == "generate":
			return attempt(call_generate_combo)

		# auto: try generate (combo) first, then chat
		out = attempt(call_generate_combo)
		if out:
			return out
		return attempt(lambda: self._chat(system, user, temperature, verbose))
