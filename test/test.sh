#!/bin/bash
x11vnc -q -bg -display $DISPLAY
python test_virtual_display.py
