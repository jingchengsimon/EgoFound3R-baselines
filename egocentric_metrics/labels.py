"""Explicit joint/vertex visibility and contact metric entry points."""

from __future__ import annotations

from .binary import binary_metrics
from .contact import contact_distance_metrics


def joint_visibility_metrics(prediction, target, mask=None, **kwargs):
    """Binary visibility metrics over MANO joints."""
    return binary_metrics(prediction, target, mask=mask, **kwargs)


def vertex_visibility_metrics(prediction, target, mask=None, **kwargs):
    """Binary visibility metrics over MANO vertices."""
    return binary_metrics(prediction, target, mask=mask, **kwargs)


def joint_contact_metrics(prediction, target, mask=None, **kwargs):
    """Binary contact metrics over MANO joints."""
    return binary_metrics(prediction, target, mask=mask, **kwargs)


def vertex_contact_metrics(prediction, target, mask=None, **kwargs):
    """Binary contact metrics over MANO vertices."""
    return binary_metrics(prediction, target, mask=mask, **kwargs)


def joint_contact_distance_metrics(prediction, target, mask=None, **kwargs):
    """Continuous contact-distance metrics over MANO joints."""
    return contact_distance_metrics(prediction, target, mask=mask, **kwargs)


def vertex_contact_distance_metrics(prediction, target, mask=None, **kwargs):
    """Continuous contact-distance metrics over MANO vertices."""
    return contact_distance_metrics(prediction, target, mask=mask, **kwargs)
