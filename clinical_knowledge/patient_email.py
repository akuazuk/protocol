"""Отправка отчётов по SMTP (опционально)."""
from __future__ import annotations

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any


def smtp_configured() -> bool:
    return bool(
        (os.environ.get("SMTP_HOST") or "").strip()
        and (os.environ.get("SMTP_FROM") or "").strip()
    )


def send_report_email(
    *,
    subject: str,
    body_text: str,
    body_html: str | None = None,
    to_addrs: list[str] | None = None,
) -> dict[str, Any]:
    """Отправить отчёт. Без SMTP - возвращает skipped."""
    default_to = (os.environ.get("PATIENT_REPORT_EMAIL_TO") or "akuazuk@gmail.com").strip()
    recipients = [a.strip() for a in (to_addrs or [default_to]) if a and a.strip()]
    if not recipients:
        return {"ok": False, "error": "no_recipients"}
    if not smtp_configured():
        return {"ok": False, "skipped": True, "reason": "smtp_not_configured"}

    host = os.environ["SMTP_HOST"].strip()
    port = int(os.environ.get("SMTP_PORT") or "587")
    user = (os.environ.get("SMTP_USER") or "").strip()
    password = (os.environ.get("SMTP_PASS") or os.environ.get("SMTP_PASSWORD") or "").strip()
    sender = os.environ["SMTP_FROM"].strip()
    use_tls = os.environ.get("SMTP_TLS", "1").strip().lower() not in ("0", "false", "no")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(body_text, "plain", "utf-8"))
    if body_html:
        msg.attach(MIMEText(body_html, "html", "utf-8"))

    with smtplib.SMTP(host, port, timeout=30) as smtp:
        if use_tls:
            smtp.starttls()
        if user and password:
            smtp.login(user, password)
        smtp.sendmail(sender, recipients, msg.as_string())
    return {"ok": True, "to": recipients}
