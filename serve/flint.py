"""serve/flint.py - Recursive Flint's text for the server.

Flint writes a tool call as (its chat template; the GLM-4.7 form)

    <tool_call>get_weather<arg_key>city</arg_key><arg_value>Zagreb</arg_value><arg_key>days</arg_key><arg_value>3</arg_value></tool_call>

string values as they are, every other type as JSON.  It decides on each turn whether to reason: a turn opens with
<think>...</think> or goes straight to the answer, and its template has no thinking switch.

The server's OutputParser (serve/frontend.py) reads the Qwen form `<tool_call><function=NAME><parameter=P>...`, with
its streaming of tool arguments, its calls inside the thinking and its recovery rules.  FlintTranslator rewrites
Flint's text into that form as it streams (holding back only a tag that is still arriving), the way serve/dsml.py
does for DeepSeek, and FlintOutputParser is the OutputParser behind it: always started in reasoning, the translator
closing the reasoning at once when the turn does not open with <think>.
"""
from __future__ import annotations

from serve.frontend import Event, OutputParser

# generation_config.json's ends of an answer: <|end_of_text|>, <|user|>, <|observation|>, <|developer|>
STOP_IDS = (150001, 150003, 150005, 150018)
THINK_OPEN = "<think>"
_TAGS = ("<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>")
_NAME_MAX = 300                 # a tool name longer than this is text, not held any further


class FlintTranslator:
    """Flint's text -> the Qwen tool-call text, a delta at a time.  Text outside a call passes through unchanged."""

    def __init__(self):
        self.buf = ""
        self.state = "start"   # start | text | name | args | key | value
        self.name = ""
        self.key = ""

    @property
    def held(self) -> str:
        return self.buf

    def feed(self, delta: str) -> str:
        self.buf += delta
        return self._run(final=False)

    def flush(self) -> str:
        out = self._run(final=True)
        if self.state in ("name", "args", "key", "value"):   # the call was cut short: close what is open
            if self.state == "name":
                out += f"<tool_call>\n<function={self.name.strip()}>\n"
            if self.state == "key":
                out += f"<parameter={self.key.strip()}>\n"
            if self.state in ("key", "value"):
                out += "\n</parameter>\n"
            out += "</function>\n</tool_call>\n"
            self.state = "text"
        return out

    def _run(self, final: bool) -> str:
        out: list[str] = []
        while self.buf:
            if self.state == "start":
                s = self.buf.lstrip()
                if not s:
                    if final:
                        self.buf = ""
                    break
                if s.startswith(THINK_OPEN):
                    self.buf = s[len(THINK_OPEN):]
                    self.state = "text"         # the parser is in the reasoning already
                    continue
                if THINK_OPEN.startswith(s) and not final:
                    break                       # may still become <think>
                out.append("</think>\n\n")      # no reasoning this turn: the answer starts now
                self.state = "text"
                continue
            if self.state == "name":
                lt = self.buf.find("<")
                if lt < 0:
                    if len(self.name) + len(self.buf) > _NAME_MAX:   # not a call after all
                        out.append("<tool_call>" + self.name + self.buf)
                        self.buf, self.name, self.state = "", "", "text"
                        continue
                    self.name += self.buf
                    self.buf = ""
                    break
                self.name += self.buf[:lt]
                self.buf = self.buf[lt:]
                if self.buf.startswith("<arg_key>") or self.buf.startswith("</tool_call>"):
                    out.append(f"<tool_call>\n<function={self.name.strip()}>\n")
                    self.state = "args"
                    continue
                if any(t.startswith(self.buf) for t in ("<arg_key>", "</tool_call>")) and not final:
                    break
                out.append("<tool_call>" + self.name)   # some other '<': this was text
                self.name, self.state = "", "text"
                continue
            if self.state == "key":
                end = self.buf.find("</arg_key>")
                if end < 0:
                    if not final:
                        keep = self._partial(self.buf, ("</arg_key>",))
                        self.key += self.buf[:len(self.buf) - keep]
                        self.buf = self.buf[len(self.buf) - keep:]
                        break
                    self.key += self.buf
                    self.buf = ""
                    break
                self.key += self.buf[:end]
                self.buf = self.buf[end + len("</arg_key>"):]
                out.append(f"<parameter={self.key.strip()}>\n")
                self.key = ""
                self.state = "after_key"
                continue
            if self.state == "after_key":
                s = self.buf.lstrip()
                if s.startswith("<arg_value>"):
                    self.buf = s[len("<arg_value>"):]
                    self.state = "value"
                    continue
                if not s or ("<arg_value>".startswith(s) and not final):
                    self.buf = s
                    break
                self.state = "value"             # the value's opening tag was left out
                continue
            if self.state == "value":
                end = self.buf.find("</arg_value>")
                if end < 0:
                    keep = 0 if final else self._partial(self.buf, ("</arg_value>",))
                    out.append(self.buf[:len(self.buf) - keep])
                    self.buf = self.buf[len(self.buf) - keep:]
                    break
                out.append(self.buf[:end])
                out.append("\n</parameter>\n")
                self.buf = self.buf[end + len("</arg_value>"):]
                self.state = "args"
                continue
            if self.state == "args":
                s = self.buf.lstrip()
                if s.startswith("<arg_key>"):
                    self.buf = s[len("<arg_key>"):]
                    self.state = "key"
                    continue
                if s.startswith("</tool_call>"):
                    self.buf = s[len("</tool_call>"):]
                    out.append("</function>\n</tool_call>\n")
                    self.state = "text"
                    continue
                if not s or (any(t.startswith(s) for t in ("<arg_key>", "</tool_call>")) and not final):
                    self.buf = s
                    break
                out.append("</function>\n</tool_call>\n")   # stray text: the call ends here
                self.state = "text"
                continue
            # text
            i = self.buf.find("<tool_call>")
            if i < 0:
                keep = 0 if final else self._partial(self.buf, ("<tool_call>",))
                out.append(self.buf[:len(self.buf) - keep])
                self.buf = self.buf[len(self.buf) - keep:]
                break
            out.append(self.buf[:i])
            self.buf = self.buf[i + len("<tool_call>"):]
            self.name, self.state = "", "name"
        return "".join(out)

    @staticmethod
    def _partial(s: str, tags) -> int:
        """How many characters at the end of `s` may be the start of one of `tags`."""
        best = 0
        for t in tags:
            for n in range(min(len(t) - 1, len(s)), 0, -1):
                if t.startswith(s[-n:]):
                    best = max(best, n)
                    break
        return best


class FlintOutputParser(OutputParser):
    """OutputParser for Flint: its calls are read as the template-form calls they are, its reasoning only where it
    writes <think>."""

    def __init__(self, *args, **kwargs):
        kwargs["thinking"] = True                 # Flint decides; the translator closes the reasoning when it does not
        if args:
            args = (True,) + tuple(args[1:])
        super().__init__(*args, **kwargs)
        self.fl = FlintTranslator()
        self._reasoning: list[str] = []
        self._answered = False

    def _watch(self, evs):
        for ev in evs:
            if ev.kind == "reasoning":
                self._reasoning.append(ev.text)
            elif ev.kind in ("tool_start", "tool_call") or (ev.kind == "content" and ev.text.strip()):
                self._answered = True
        return evs

    def feed(self, delta: str):
        return self._watch(super().feed(self.fl.feed(delta)))

    def finish(self, reason: str | None = None):
        rest = self.fl.flush()
        out = self._watch(super().feed(rest) if rest else [])
        out += self._watch(super().finish(reason))
        text = "".join(self._reasoning).strip()
        if reason in (None, "stop") and text and not self._answered:
            # a <think> that never closed and no answer: the text is the answer (as serve/dsml.py does)
            self._answered = True
            out.append(Event("content", text))
        return out

    @property
    def clean(self) -> bool:
        return not self.buf and not self.fl.held
