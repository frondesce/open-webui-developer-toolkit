# Changelog

All notable changes to the User Map Context Clip Filter are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.2] - 2025-12-23
- Change limit semantics to count only context messages (current message excluded).
- Allow `0` in user limits to mean "no context".

## [0.7.1] - 2025-12-12
- Initial release: per-user context clipping by ID/email, admin unlimited, optional system preservation.
