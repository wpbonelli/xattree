from typing import Optional

import pytest

from xattree import field, xattree


@xattree
class Child:
    pass


# An explicit name should stick, through every attachment path, for both
# `only`- and `list`-kind fields. (`dict`-kind already honors given name


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


def test_only_kind_getattr_finds_explicitly_named_child_via_parent_attach():
    @xattree
    class Parent:
        child: Optional[Child] = field(default=None)

    parent = Parent()
    child = Child(name="custom", parent=parent)
    assert parent.child is child


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
