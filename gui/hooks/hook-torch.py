# -----------------------------------------------------------------------------
# Copyright (c) 2013-2018, PyInstaller Development Team.
#
# Distributed under the terms of the GNU General Public License with exception
# for distributing bootloader.
#
# The full license is in the file COPYING.txt, distributed with this software.
# -----------------------------------------------------------------------------

from PyInstaller.utils.hooks import get_package_paths
from PyInstaller.utils.hooks.conda import collect_dynamic_libs

datas = [(get_package_paths('torch')[1], "torch"), ]

binaries = collect_dynamic_libs('pytorch')
