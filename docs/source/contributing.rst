Contributing
============

.. role:: bash(code)
    :language: bash

.. role:: cmd(code)
    :language: bash

TorchUncertainty is in an early development stage. We are looking for
contributors to help us build a comprehensive library for uncertainty
quantification in PyTorch.

We are particularly open to any comment you may have on this project, and we are open to
revising these guidelines as the project evolves.

The scope of TorchUncertainty
-----------------------------

TorchUncertainty can host any method — ideally linked to a paper — that fits in one of the
following fields:

* uncertainty quantification in general, including Bayesian deep learning, Monte Carlo dropout, ensemble methods, etc.;
* out-of-distribution detection methods;
* applications (e.g., object detection, segmentation, depth estimation, etc.).

Common guidelines
-----------------

Clean development install of TorchUncertainty
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are interested in contributing to TorchUncertainty, we recommend the
following steps to set up a clean development environment and ensure that the
continuous integration does not break.

1. Install ``uv`` following the steps `here <https://docs.astral.sh/uv/getting-started/installation/>`_
2. Clone the repository
3. Install torch-uncertainty with the dev packages:
   :cmd:`uv sync --extra gpu` for GPU-based systems or
   :cmd:`uv sync --extra cpu` if no GPUs are available

.. note::

    Failure to include the extra flag will result in the GPU version of PyTorch without CUDA and might cause issues.

4. Install pre-commit hooks with :cmd:`uv run pre-commit install`

Build the documentation locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build the documentation on Linux, navigate to ``./docs`` and build the documentation with:

.. parsed-literal::

    make html

Optionally, specify ``html-noplot`` instead of ``html`` to avoid running the tutorials.
This option is necessary if you only have a CPU on your machine.

Guidelines
^^^^^^^^^^

**Code quality**

We use ruff for code formatting, linting, and import sorting (as a drop-in
replacement for black, isort, and flake8). The pre-commit hooks will ensure
that your code is properly formatted and linted before committing.

To check that your code complies with the standards, run the following and address any warnings:

.. parsed-literal::

    uv run ruff check --fix

And then:

.. parsed-literal::

    uv run ruff format

Please ensure that the tests pass on your machine before pushing a PR. This avoids
adding featureless commits just to chase CI. To run the tests, from the root of the
repository:

.. parsed-literal::

    uv run pytest tests

**Commit message convention**

We follow a structured commit message format inspired by
`gitmoji <https://gitmoji.dev/>`_ and the
`Conventional Commits <https://www.conventionalcommits.org/>`_ specification:

.. code-block:: text

    :emoji:(scope): description

* **Emoji**: pick the most appropriate emoji from `gitmoji <https://gitmoji.dev/>`_
  to categorize the change at a glance.
* **Scope**: a short identifier in parentheses describing the area of the codebase
  affected (e.g. ``packed``, ``metrics``, ``docs``, ``CI``, ``all``).
* **Description**: a concise summary of the change after the ``:`` separator.
* **Breaking changes**: use ``!`` before ``:`` to signal a breaking change.

Here are some examples from the project:

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Commit message
     - Meaning
   * - ``:bug:(cli): fix self.model not being a property``
     - Bug fix in the CLI module
   * - ``:sparkles:(losses): add the DER loss``
     - New feature in losses
   * - ``:recycle:(segformer): remove init redundancies``
     - Refactoring of segformer
   * - ``:rotating_light:(all): ruff format``
     - Fixing linter warnings across the codebase
   * - ``:memo:(tabular): add some documentation``
     - Documentation for the tabular module
   * - ``:white_check_mark:(coverage): fix coverage misses``
     - Test improvements
   * - ``:wrench:(uv): update uv & workflows``
     - Configuration/tooling changes
   * - ``:boom:(routines)!: rename eval to test``
     - Breaking change in routines

Some of the most commonly used gitmoji in this project:

.. list-table::
   :header-rows: 1
   :widths: 10 25 65

   * - Emoji
     - Code
     - Use
   * - ✨
     - ``:sparkles:``
     - New feature
   * - 🐛
     - ``:bug:``
     - Bug fix
   * - ♻️
     - ``:recycle:``
     - Refactor code
   * - 📝
     - ``:memo:``
     - Documentation
   * - 🚨
     - ``:rotating_light:``
     - Fix linter warnings
   * - ✅
     - ``:white_check_mark:``
     - Add/fix tests
   * - 🔧
     - ``:wrench:``
     - Configuration files
   * - 🔥
     - ``:fire:``
     - Remove code or files
   * - ⬆️
     - ``:arrow_up:``
     - Upgrade dependencies
   * - 💚
     - ``:green_heart:``
     - Fix CI build
   * - 👷
     - ``:construction_worker:``
     - CI system changes
   * - 🎨
     - ``:art:``
     - Improve code structure/format
   * - 💥
     - ``:boom:``
     - Breaking changes
   * - 👌
     - ``:ok_hand:``
     - Code review changes

For the full list of available emojis, visit `gitmoji.dev <https://gitmoji.dev/>`_.

You don't need to follow this convention.

**Pull requests**

To make your changes, create a branch on a personal fork and create a PR when your contribution
is mostly finished or if you need help.

*PR naming convention*

Pull requests should follow the same naming convention as commits:

.. code-block:: text

    :emoji:(scope): description

For instance: ``:sparkles:(datamodules): add support for CIFAR-100-C``.

*PR labels*

Please flag your PRs with the appropriate labels to help maintainers triage and
track changes. Common labels include:

* ``enhancement`` — new feature or improvement
* ``bug`` — bug fix
* ``documentation`` — documentation changes
* ``refactor`` — code restructuring without behavior change
* ``tests`` — test additions or fixes
* ``dependencies`` — dependency updates
* ``breaking-change`` — introduces a breaking change
* ``need-help`` — you need assistance from a maintainer

*PR checklist*

Check that your PR complies with the following conditions:

* The name of your branch is not ``main`` nor ``dev`` (see issue #58)
* Your PR does not reduce the code coverage
* Your code is documented: the function signatures are typed, and the main functions have clear docstrings
* Your code is mostly original, and the parts coming from licensed sources are explicitly stated as such
* If you implement a method, please add a reference to the corresponding paper in the
  `references page <https://torch-uncertainty.github.io/references.html>`_.
* If you implement a metric that you add to one of the routines, add a reference in the 
  `evaluation page <https://torch-uncertainty.github.io/evaluation.html>`_.

If you need help to implement a method, increase the coverage, or solve ruff-raised errors,
create the PR with the ``need-help`` flag and explain your problems in the comments. A maintainer
will do their best to help you.

Datasets & Datamodules
^^^^^^^^^^^^^^^^^^^^^^

We intend to include datamodules for the most popular datasets only.

Post-processing methods
^^^^^^^^^^^^^^^^^^^^^^^

For now, we intend to follow scikit-learn style API for post-processing
methods (except that we use a validation dataset instead of a numpy array).
You may get inspiration from the already implemented
`temperature-scaling <https://github.com/torch-uncertainty/torch-uncertainty/blob/dev/torch_uncertainty/post_processing/calibration/temperature_scaler.py>`_.


License
-------

If you feel that the current license is an obstacle to your contribution, let
us know, and we may reconsider. However, the models' weights hosted on Hugging
Face are likely to remain Apache 2.0.
