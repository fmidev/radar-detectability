# SPDX-FileCopyrightText: 2026-present Jussi Tiira <jussi.tiira@fmi.fi>
#
# SPDX-License-Identifier: MIT

"""Installed package version, derived from Git tags by hatch-vcs at build time."""

from importlib.metadata import version

DISTRIBUTION = "radar-detectability"
__version__ = version(DISTRIBUTION)
