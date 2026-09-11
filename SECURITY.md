# Security Policy

## Supported version

Repository ini merupakan prototipe riset. Perbaikan keamanan diterapkan pada branch `main` terbaru.

## Deployment boundary

Server tidak menyediakan autentikasi dan hanya dirancang untuk jaringan lokal tepercaya. Jangan mengekspos port `8080` langsung ke internet. Gunakan firewall untuk membatasi akses dan, bila akses jarak jauh diperlukan, tempatkan sistem di belakang reverse proxy yang menyediakan TLS dan autentikasi.

Endpoint sistem dapat membawa frame kamera, audio, transkripsi, dan perintah kontrol. Perlakukan seluruh output runtime pada `logs/`, `layout.json`, `debug_*.jpg`, dan `tts_cache/` sebagai data privat. Artefak tersebut sudah dikecualikan dari Git.

## Reporting a vulnerability

Laporkan kerentanan secara privat melalui GitHub Security Advisories pada repository ini. Jangan menyertakan frame kamera, rekaman suara, transkripsi pengguna, kredensial, atau data pribadi dalam issue publik.
