"""
Tests for the child-naming consolidation: a caller-given `name=` persists
as the real attachment key (and therefore `.name`) for all three child
kinds (`only`, `list`, `dict`), through every attachment path (a
constructor kwarg, `parent=`, or plain attribute assignment). Collisions
are checked across a parent's *whole* child namespace (not just siblings
from the same field): an explicit-name collision raises, while an
auto-generated (unnamed) name never collides in the first place -- it
picks the next free slot instead. Unnamed default naming is
field-name-based for every attachment path.
"""

from typing import Optional

import pytest

from xattree import field, xattree


@xattree
class Child:
    pass


# ---------------------------------------------------------------------------
# An explicit name should stick, through every attachment path, for both
# `only`- and `list`-kind fields. (`dict`-kind already honors the given
# name -- that's the pattern this plan generalizes.)
# ---------------------------------------------------------------------------


def test_only_kind_explicit_name_sticks_kwarg_attach():
    @xattree
    class Parent:
        child: Child = field()

    child = Child(name="custom")
    parent = Parent(child=child)
    assert child.name == "custom"
    assert "custom" in parent.data.children


def test_list_kind_explicit_name_sticks_kwarg_attach():
    @xattree
    class Parent:
        child_list: list[Child] = field()

    child = Child(name="custom")
    parent = Parent(child_list=[child])
    assert child.name == "custom"
    assert "custom" in parent.data.children


def test_only_kind_explicit_name_sticks_parent_attach():
    """Already honored by `_bind_tree` today (once the parent-truthiness fix landed)."""

    @xattree
    class Parent:
        child: Optional[Child] = field(default=None)

    parent = Parent()
    child = Child(name="custom", parent=parent)
    assert child.name == "custom"
    assert "custom" in parent.data.children


def test_list_kind_explicit_name_sticks_parent_attach():
    @xattree
    class Parent:
        child_list: list[Child] = field()

    parent = Parent()
    child = Child(name="custom", parent=parent)
    assert child.name == "custom"


# ---------------------------------------------------------------------------
# `only`-kind lookup must be able to find a child attached under its own
# explicit name, not just under the field name.
# ---------------------------------------------------------------------------


def test_only_kind_getattr_finds_explicitly_named_child_via_parent_attach():
    @xattree
    class Parent:
        child: Optional[Child] = field(default=None)

    parent = Parent()
    child = Child(name="custom", parent=parent)
    assert parent.child is child


# ---------------------------------------------------------------------------
# Unnamed default naming is field-name-based for both attachment paths.
# Path 2 (parent=-attach) used to default to the bare class name instead,
# ignoring the field name entirely -- a divergence from path 1 flagged in
# the plan's current-state notes, removed by consolidating onto one
# shared helper (decision 4).
# ---------------------------------------------------------------------------


def test_only_kind_unnamed_default_kwarg_attach():
    @xattree
    class Parent:
        kid: Child = field()

    child = Child()
    Parent(kid=child)
    assert child.name == "kid"


def test_list_kind_unnamed_default_kwarg_attach():
    @xattree
    class Parent:
        kids: list[Child] = field()

    a, b = Child(), Child()
    Parent(kids=[a, b])
    assert a.name == "kids0"
    assert b.name == "kids1"


def test_only_kind_unnamed_default_parent_attach():
    @xattree
    class Parent:
        kid: Optional[Child] = field(default=None)

    parent = Parent()
    child = Child(parent=parent)
    assert child.name == "kid"


def test_list_kind_unnamed_default_parent_attach():
    @xattree
    class Parent:
        kids: list[Child] = field()

    parent = Parent()
    a = Child(parent=parent)
    b = Child(parent=parent)
    assert a.name == "kids0"
    assert b.name == "kids1"


# ---------------------------------------------------------------------------
# Collisions must raise immediately, not silently overwrite -- across a
# parent's whole child namespace, not just within one field.
# ---------------------------------------------------------------------------


def test_dict_kind_cross_field_explicit_name_collision_raises():
    @xattree
    class Parent:
        a: dict[str, Child] = field()
        b: dict[str, Child] = field()

    with pytest.raises(ValueError):
        Parent(a={"same": Child(name="same")}, b={"same": Child(name="same")})


def test_only_kind_cross_field_explicit_name_collision_raises():
    @xattree
    class ChildA:
        pass

    @xattree
    class ChildB:
        pass

    @xattree
    class Parent:
        a: Optional[ChildA] = field(default=None)
        b: Optional[ChildB] = field(default=None)

    parent = Parent()
    ChildA(name="same", parent=parent)
    with pytest.raises(ValueError):
        ChildB(name="same", parent=parent)


def test_auto_generated_name_does_not_collide_with_explicit_name_cross_field():
    """
    An auto-generated (unnamed) name must never collide with an explicit
    name from a *different* field -- not by raising, but by construction:
    it should pick the next free slot instead, per decision 3 ("an
    auto-generated name never collides ... by construction").
    """

    @xattree
    class ChildA:
        pass

    @xattree
    class ChildB:
        pass

    @xattree
    class Parent:
        single: Optional[ChildA] = field(default=None)
        many: list[ChildB] = field()

    parent = Parent()
    single = ChildA(name="many0", parent=parent)
    parent.many = [ChildB()]

    assert parent.single is single
    assert parent.many[0].name == "many1"
    assert set(parent.data.children) == {"many0", "many1"}
