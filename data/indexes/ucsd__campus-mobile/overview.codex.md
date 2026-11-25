## What it is

`UCSD/campus-mobile` appears to be a Flutter app for UC San Diego students and campus users. Its screens and providers cover campus information and services such as dining, maps, parking, shuttles, Wi-Fi, notifications, and student and employee ID cards; it also includes an AI assistant.

## How it’s organized

- `lib/` contains the Dart application. Its top level includes startup, routing, networking, constants, and styles; `lib/core/` holds models, providers, services, utilities, and wrappers; `lib/ui/` holds feature screens and components.
- `lib/ui/` is divided by feature, including `lib/ui/dining/`, `lib/ui/map/`, `lib/ui/parking/`, `lib/ui/shuttle/`, `lib/ui/ai_assistant/`, and `lib/ui/profile/`.
- `android/` and `ios/` contain the platform-specific app projects and assets.
- `test/` contains tests organized under `core/` and `ui/`.
- `scripts/` contains formatting and naming checks, install support, and Codemagic CI helpers. `.github/` contains issue and pull request templates plus a pre-commit workflow.

## How it works

1. `lib/main.dart` defines `main()` and initialization helpers, including `initializeHive()` and `initializeApp()`. This appears to be the app’s startup path.
2. `lib/app_router.dart` and `lib/app_constants.dart` appear to define navigation and route names. The supplied outlines do not show the router’s full behavior.
3. Feature screens in `lib/ui/` display app content. For example, `lib/ui/home/home.dart` builds the home screen and ordered cards, while `lib/ui/dining/dining_detail_view.dart` builds a dining hall detail view.
4. Providers in `lib/core/providers/` manage feature data and actions. For instance, `lib/core/providers/user.dart` handles login state, and `lib/core/providers/cards.dart` loads and updates home-card state and order.
5. `lib/app_networking.dart` defines fetch and authorized request helpers, while feature providers and services appear to use these for network-backed data. The supplied outlines don’t establish the exact request-to-service flow for each feature.

## Where to start reading

1. `lib/main.dart` — see how the app starts and initializes local storage.
2. `lib/app_router.dart` — inspect the app’s navigation entry points.
3. `lib/app_constants.dart` — find route names and shared feature constants.
4. `lib/ui/home/home.dart` — see how the home screen assembles cards.
5. `lib/core/providers/user.dart` — follow authentication and persisted user state.
6. `lib/app_networking.dart` — review the shared network request methods.

## Tech stack

Dart and Flutter appear to make up the main app stack; the repository also includes Android Gradle/Kotlin configuration, an iOS Swift app delegate, and Bash, Python, JavaScript, and Groovy support files.

## Running it

It isn’t documented in the supplied material. The file tree includes `pubspec.yaml` and platform projects, but no install or run commands were provided.