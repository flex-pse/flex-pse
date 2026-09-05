flexops.logic
=============

This is the composable unit commitment layer, a set of optional
constraint pieces applied per unit. :func:`~flexops.logic.status.add_status` is
the base (present whenever a unit can be shut off), and everything else is
opt-in. Every name below is re-exported from ``flexops.logic`` itself.

Status
------

.. currentmodule:: flexops.logic.status

.. autofunction:: add_status

.. autofunction:: relax

.. autofunction:: unrelax

Commitment and timing
---------------------

.. currentmodule:: flexops.logic.unit_commitment

.. autofunction:: add_startup_shutdown

.. currentmodule:: flexops.logic.delays

.. autofunction:: add_startup_delay

.. currentmodule:: flexops.logic.dwell

.. autofunction:: add_dwell

``add_dwell`` is a distinct, unrelated concept from ``add_startup_shutdown``'s
minimum uptime/downtime. It holds a **continuous** process variable steady, not
a unit's on/off status.

Ramp rate
---------

.. currentmodule:: flexops.logic.ramp

.. autofunction:: add_ramp_rate

Bypass and implications
-----------------------

.. currentmodule:: flexops.logic.bypass

.. autofunction:: add_bypass

.. currentmodule:: flexops.logic.conditional

.. autofunction:: add_conditional

Parallel trains and degeneracy
------------------------------

.. currentmodule:: flexops.logic.degeneracy

Symmetry among identical parallel trains creates degeneracy in solver time.
Many solutions with equal objective values differ only in *which*
interchangeable unit is on, or in how a shared duty is split between them.
A unit cannot see its siblings, so this is handled **outside the unit level**.
The caller declares the group, not a unit's own ``build()``.

The group is listed in priority order and ordered **descending** along that
list::

    group[0].status[t] >= group[1].status[t] >= ... >= group[-1].status[t]

so a train may not run unless its predecessor runs, and the unit listed first
is the first one on. The same ordering applies to any continuous Vars named in
``variables``. What an ordering buys is a smaller search, since it cuts the
permuted copies of a split duty, rather than a unique split. A split with an
equal cost that is not merely a relabelling of another still satisfies it.

Status ordering is the default and requires a ``status`` Var on every unit in
the group. Since ``status`` is attached by the caller
(:func:`~flexops.logic.status.add_status`) rather than by a unit's ``build()``,
a group that never had unit commitment applied (or one made up of units that
have no on/off state at all, such as a tank) is registered with
``order_status=False``, which orders the named ``variables`` alone.

.. autofunction:: register_parallel_group
