"""Fixed bench camera only: no task, object, predicate or simulator-state lookup."""
import xml.etree.ElementTree as ET

import numpy as np

NAME = 'oppositeview'
POSITION = (-0.05651774593317116, -1.2761224129427358, 1.4879572214102434)
LOOK_AT = (0., 0., .8)
FOVY = 45.


def camera_attributes():
    forward = np.asarray(LOOK_AT)-np.asarray(POSITION)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return dict(name=NAME, mode='fixed', pos=' '.join(map(str, POSITION)),
                xyaxes=' '.join(map(str, np.r_[right, up])), fovy=str(FOVY))


def append_fixed_camera(xml):
    """Append one world camera without reserializing any existing XML bytes."""
    root = ET.fromstring(xml)
    if root.find('worldbody') is None or xml.count('</worldbody>') != 1:
        raise ValueError('Expected one native worldbody')
    existing = [c for c in root.iter('camera') if c.get('name') == NAME]
    attributes = camera_attributes()
    if existing:
        if len(existing) != 1 or existing[0].attrib != attributes:
            raise ValueError('Fixed recovery camera name already has another definition')
        return xml
    camera = ET.tostring(ET.Element('camera', attributes), encoding='unicode')
    return xml.replace('</worldbody>', camera+'\n</worldbody>', 1)
