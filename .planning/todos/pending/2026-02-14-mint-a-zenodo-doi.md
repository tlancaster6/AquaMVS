---
created: 2026-02-14T19:14:41.511Z
title: Mint a Zenodo DOI
area: general
files: []
---

## Problem

AquaMVS needs a citable DOI so researchers can reference it in publications. Zenodo integration with GitHub can automatically mint DOIs for releases, providing a persistent identifier for each version.

## Solution

1. Link the AquaMVS GitHub repository to Zenodo (via zenodo.org GitHub integration)
2. Create a `.zenodo.json` metadata file with authors, title, description, license, keywords
3. On next GitHub release, Zenodo will automatically mint a DOI
4. Add the DOI badge to README.md
