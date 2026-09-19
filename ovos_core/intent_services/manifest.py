# Copyright 2024 OpenVoiceOS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ovos_bus_client.message import Message
from typing import Iterable, Optional

from ovos_spec_tools import closest_lang, standardize_lang
from ovos_utils.log import LOG


class IntentManifest:
    """INTENT-4 §10 orchestrator-owned manifest.

    Indexes every ``ovos.intent.register.*`` broadcast and serves
    ``ovos.intent.list`` / ``ovos.intent.describe`` pull-queries.
    The manifest is keyed by the quintuple
    ``(session_id, skill_id, intent_name, lang, method)`` per §11.1.

    ``ovos.intent.list`` accepts ``{"include_definitions": true}``: each
    result row then also carries ``definition``, the stored registration
    payload that ``ovos.intent.describe`` would return for it, so a client
    that wants every intent's sentences needs one round trip per language
    instead of one describe per intent. Without the flag the reply is the
    plain §10.1 row.

    A ``lang`` asked of any query is resolved per intent the way utterance
    matching resolves it (see :meth:`_resolve_lang`): an intent registered
    only as ``en-US`` is listed and described for an ``en-CA`` client, and
    its rows keep their registered ``lang`` so the client sees what it got.
    """

    def __init__(self, bus):
        self.bus = bus
        # (session_id, skill_id, intent_name, lang, method) → entry dict
        self._index: dict = {}

        bus.on("ovos.intent.register.keyword", self._on_register)
        bus.on("ovos.intent.register.template", self._on_register)
        bus.on("ovos.intent.deregister", self._on_deregister)
        bus.on("ovos.intent.enable", self._on_enable_disable)
        bus.on("ovos.intent.disable", self._on_enable_disable)
        bus.on("ovos.skill.deregister", self._on_skill_deregister)
        bus.on("ovos.intent.list", self._on_list)
        bus.on("ovos.intent.describe", self._on_describe)

    def shutdown(self):
        self.bus.remove("ovos.intent.register.keyword", self._on_register)
        self.bus.remove("ovos.intent.register.template", self._on_register)
        self.bus.remove("ovos.intent.deregister", self._on_deregister)
        self.bus.remove("ovos.intent.enable", self._on_enable_disable)
        self.bus.remove("ovos.intent.disable", self._on_enable_disable)
        self.bus.remove("ovos.skill.deregister", self._on_skill_deregister)
        self.bus.remove("ovos.intent.list", self._on_list)
        self.bus.remove("ovos.intent.describe", self._on_describe)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(session_id: str, skill_id: str, intent_name: str,
              lang: str, method: str) -> tuple:
        return session_id, skill_id, intent_name, standardize_lang(lang), method

    @staticmethod
    def _resolve_lang(lang: str, registered: Iterable[str]) -> Optional[str]:
        """The registered language that answers a query in *lang*.

        The same OVOS-INTENT-2 §2.2 fallback utterance matching applies
        (``ovos_spec_tools.closest_lang``, which the adapt, padatious and
        padacioso engines and ovos-m2v-pipeline's per-label resolution all
        call): an exact match wins, otherwise the nearest registered region,
        so ``en-CA`` reaches ``en-US`` and ``fr-CA`` reaches ``fr-FR``.

        Candidates are narrowed to *lang*'s primary subtag first. The
        language distance alone would admit a neighbouring language
        (``bs`` against ``hr`` is within the threshold); a listing never
        answers in a language the client did not ask for.

        Returns ``None`` when nothing registered is close enough.
        """
        lang = standardize_lang(lang)
        primary = lang.split("-")[0].lower()
        candidates = sorted({r for r in registered
                             if r.split("-")[0].lower() == primary})
        return closest_lang(lang, candidates)

    @classmethod
    def _answering_langs(cls, entries: list, lang: str) -> dict:
        """Map each ``(skill_id, intent_name)`` in *entries* to the
        registered language answering *lang*, resolved per intent the way
        ovos-m2v-pipeline resolves each label: an intent registered in a
        single dialect keeps it whatever dialects other skills use."""
        by_intent: dict = {}
        for entry in entries:
            by_intent.setdefault((entry["skill_id"], entry["intent_name"]),
                                 set()).add(entry["lang"])
        return {intent: cls._resolve_lang(lang, langs)
                for intent, langs in by_intent.items()}

    def _effective_pool(self, session_id: str) -> list:
        """Return entries for *session_id* merged with 'default' (§11.2)."""
        seen = {}
        for key, entry in self._index.items():
            s, skill, name, lang, method = key
            if s not in ("default", session_id):
                continue
            dedup = (skill, name, lang, method)
            if dedup not in seen or s == session_id:
                seen[dedup] = entry
        return list(seen.values())

    def get_required_slots(self, session_id: str, skill_id: str,
                           intent_name: str, lang: str) -> list:
        """OVOS-INTENT-4 §6.1 / §10 — the ``required_slots`` an intent declares.

        The canonical source for the OVOS-PIPELINE-1 §6.2 orchestrator backstop:
        the required-slot names an intent registered under its
        ``ovos.intent.register.*`` payload. Merges the union across the intent's
        keyword/template registrations in the session's effective pool (§11.2).
        Returns ``[]`` when the intent is not in the manifest (e.g. registered via
        a legacy in-process path), leaving engine-side enforcement authoritative.
        """
        entries = [e for e in self._effective_pool(session_id)
                   if e["skill_id"] == skill_id and e["intent_name"] == intent_name]
        answering = self._resolve_lang(lang, {e["lang"] for e in entries})
        slots: list = []
        for entry in entries:
            if entry["lang"] != answering:
                continue
            for slot in (entry.get("definition") or {}).get("required_slots") or []:
                if slot not in slots:
                    slots.append(slot)
        return slots

    # ------------------------------------------------------------------
    # registration broadcasts  §§5–8
    # ------------------------------------------------------------------

    def _on_register(self, message: Message):
        method = "keyword" if message.msg_type == "ovos.intent.register.keyword" else "template"
        skill_id = message.data.get("skill_id") or message.context.get("skill_id")
        intent_name = message.data.get("intent_name")
        lang = message.data.get("lang")
        if not (skill_id and intent_name and lang):
            LOG.warning(f"malformed intent registration from {skill_id!r}: missing required fields")
            return
        session_id = (message.context.get("session") or {}).get("session_id", "default")
        key = self._key(session_id, skill_id, intent_name, lang, method)
        self._index[key] = {
            "skill_id": skill_id,
            "intent_name": intent_name,
            "lang": standardize_lang(lang),
            "method": method,
            "enabled": True,
            "session_id": session_id,
            "definition": message.data,
        }

    def _on_deregister(self, message: Message):
        skill_id = message.data.get("skill_id") or message.context.get("skill_id")
        intent_name = message.data.get("intent_name")
        lang = message.data.get("lang")
        session_id = message.data.get("session_id", "default")
        if not (skill_id and intent_name):
            return
        for method in ("keyword", "template"):
            if lang:
                self._index.pop(self._key(session_id, skill_id, intent_name, lang, method), None)
            else:
                for key in [k for k in self._index
                            if k[0] == session_id and k[1] == skill_id
                            and k[2] == intent_name and k[4] == method]:
                    del self._index[key]

    def _on_enable_disable(self, message: Message):
        enabled = message.msg_type == "ovos.intent.enable"
        skill_id = message.data.get("skill_id") or message.context.get("skill_id")
        intent_name = message.data.get("intent_name")
        lang = message.data.get("lang")
        session_id = message.data.get("session_id", "default")
        for key, entry in self._index.items():
            if key[0] != session_id or key[1] != skill_id or key[2] != intent_name:
                continue
            if lang and key[3] != standardize_lang(lang):
                continue
            entry["enabled"] = enabled

    def _on_skill_deregister(self, message: Message):
        skill_id = message.data.get("skill_id") or message.context.get("skill_id")
        session_id = message.data.get("session_id", "default")
        if not skill_id:
            return
        for key in [k for k in self._index if k[0] == session_id and k[1] == skill_id]:
            del self._index[key]

    # ------------------------------------------------------------------
    # introspection queries  §10
    # ------------------------------------------------------------------

    def _on_list(self, message: Message):
        f_skill = message.data.get("skill_id")
        f_lang = message.data.get("lang")
        f_session = message.data.get("session_id")
        # Opt-in: attach each row's registration payload (what
        # ``ovos.intent.describe`` returns as ``definition``) so a client
        # gets the sentences for a whole language in this one reply.
        include_definitions = bool(message.data.get("include_definitions", False))

        pool = self._effective_pool(f_session) if f_session else list(self._index.values())
        if f_skill:
            pool = [e for e in pool if e["skill_id"] == f_skill]
        answering = self._answering_langs(pool, f_lang) if f_lang else None
        results = []
        for entry in pool:
            if answering is not None and \
                    entry["lang"] != answering[(entry["skill_id"], entry["intent_name"])]:
                continue
            row = {k: entry[k] for k in
                   ("skill_id", "intent_name", "lang", "method", "enabled", "session_id")}
            if include_definitions:
                row["definition"] = entry["definition"]
            results.append(row)

        self.bus.emit(message.reply("ovos.intent.list.response", {"ok": True, "intents": results}))

    def _on_describe(self, message: Message):
        skill_id = message.data.get("skill_id")
        intent_name = message.data.get("intent_name")
        lang = message.data.get("lang")
        method_filter = message.data.get("method")
        session_id = message.data.get("session_id", "default")
        if not (skill_id and intent_name and lang):
            self.bus.emit(message.reply("ovos.intent.describe.response",
                                        {"ok": False,
                                         "error": "skill_id, intent_name and lang are required"}))
            return
        entries = [e for e in self._effective_pool(session_id)
                   if e["skill_id"] == skill_id and e["intent_name"] == intent_name]
        answering = self._resolve_lang(lang, {e["lang"] for e in entries})
        definitions = []
        for entry in entries:
            if entry["lang"] != answering:
                continue
            if method_filter and entry["method"] != method_filter:
                continue
            definitions.append({"method": entry["method"], "definition": entry["definition"]})
        definitions.sort(key=lambda d: 0 if d["method"] == "keyword" else 1)
        if definitions:
            self.bus.emit(message.reply("ovos.intent.describe.response",
                                        {"ok": True, "definitions": definitions}))
        else:
            self.bus.emit(message.reply("ovos.intent.describe.response",
                                        {"ok": False,
                                         "error": f"unknown intent {skill_id}:{intent_name}:{lang}"}))
