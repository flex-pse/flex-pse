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

Counted trains
~~~~~~~~~~~~~~

Pass ``max_count=n`` to :func:`~flexops.logic.status.add_status` when one unit
stands for ``n`` identical trains with the same limits. ``status[t]`` becomes
an integer count in ``0..n`` and the output limits apply per running train.
With ``min_output = 10``, ``max_output = 100`` and two trains running, the unit
can deliver between 20 and 200. One count instead of ``n`` binaries means the
solver doesn't search through equivalent orderings of the same trains. Trains
with different limits have to stay separate units.

``min_count_on=m`` stops the unit from running between 1 and ``m - 1``
trains. It adds a binary ``status_on[t]`` that's 1 when any train runs, with
``m * status_on[t] <= status[t] <= n * status_on[t]``.

:func:`~flexops.logic.unit_commitment.add_startup_shutdown` counts the trains
started and stopped at each step, and ``min_downtime`` compares against
``n - status[t]`` instead of ``1 - status[t]``. :func:`relax` turns a count
into a real in ``[0, n]``. It only changes variable domains, never
constraints, bounds or fixed values, so warm starts and references held
elsewhere survive. With the default ``max_count=1`` you get the same binary
model as before.

``add_conditional``, ``add_startup_delay`` and
``register_parallel_group(order_status=True)`` compare status to 0 and 1,
which means something different for a count (``1 - status`` can go negative).
They raise ``FlexConfigError`` on a counted unit instead of building the wrong
model.

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

Spec records
------------

.. currentmodule:: flexops.logic._records

Every helper above except ``relax`` and ``unrelax`` records the spec element
that rebuilds it: unit-level helpers on the unit, cross-unit helpers on the
model. ``emit_model`` writes those records back. A helper called inside another
(``register_parallel_group``'s conditionals, ``BatteryModel``'s own status)
runs under :func:`suppress_logic_records`, so only the outer call is written.

.. autoclass:: LogicRecord

.. autofunction:: record

.. autofunction:: suppress_logic_records
