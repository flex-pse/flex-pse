# Glossary

```{glossary}
zero-degree-of-freedom regression
  The condition FlexParameterize validates before it fits. The registered
  IO pairs exactly determine the regression problem, with no unmapped
  inputs and no free parameters left after the fit. A unit whose data
  leaves an input unmapped, or whose fit would leave a parameter free, is
  not a zero degree of freedom problem, and
  {py:func}`~flexparameterize.validate.check_sufficiency` reports it as
  insufficient.

model alias
  The dotted `plant.unit.variable` name FlexParameterize uses to refer to a
  built model's variables. It's a variable's fully qualified Pyomo name,
  for example `facility.pump.power_electrical`. A
  {py:class}`~flexparameterize.tags.TagMap` maps a data source's own
  column names onto these.
```
