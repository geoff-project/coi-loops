Optimization Loops for the Common Optimization Interfaces
=========================================================

CernML is the project of bringing numerical optimization, machine learning and
reinforcement learning to the operation of the CERN accelerator complex.

[CernML-COI][] defines common interfaces that facilitate using numerical
optimization and reinforcement learning (RL) on the same optimization problems.
This makes it possible to unify both approaches into a generic optimization
application in the CERN Control Center.

CernML-COI-Loops defines an abstract optimizer interface for all numerical
optimization algorithms. it provides implementations for a few common
third-party packages. It finally provides facilities to hook up these
optimizers to optimization problems defined via the [COI][CernML-COI].

This repository can be found online on CERN's [Gitlab][].

[Gitlab]: https://gitlab.cern.ch/geoff/cernml-coi-loops/
[CernML-COI]: https://gitlab.cern.ch/geoff/cernml-coi/

Table of Contents
-----------------

[[_TOC_]]

Motivation
----------

TODO

Quickstart
----------

TODO

Stability
---------

This package uses a variant of [Semantic Versioning](https://semver.org/) that
makes additional promises during the initial development (major version 0):
whenever breaking changes to the public API are published, the first non-zero
version number will increase. This means that code that uses COI version 0.6.0
will continue to work with version 0.6.1, but may break with version 0.7.0.

The exception to this are the contents of `cernml.coi.unstable`, which may
change in any given release.

Changelog
---------

TODO

Documentation
-------------

TODO
