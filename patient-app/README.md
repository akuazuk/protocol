# B2C Patient App (Capacitor)

Оболочка для публикации PWA в App Store / Google Play.

## Быстрый старт

```bash
cd patient-app
npm install
npx cap add ios    # или android
npx cap sync
npx cap open ios
```

`webDir` указывает на корень репозитория; в проде `server.url` - deployed `patient.html`.

Push-напоминания «обсудить с врачом» - через `@capacitor/push-notifications` (Wave C stub).
