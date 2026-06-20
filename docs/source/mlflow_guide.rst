MLflow Guide
============

TorchUncertainty uses `MLflow <https://mlflow.org>`_ as its experiment tracking backend.
All metrics, hyperparameters, and figures logged during training and evaluation are stored
in an MLflow tracking store and can be explored through the MLflow UI.

Logger configuration
--------------------

The logger is set inside the ``trainer`` block of your configuration file:

.. code:: yaml

    trainer:
      logger:
        class_path: lightning.pytorch.loggers.MLFlowLogger
        init_args:
          experiment_name: my_experiment
          tracking_uri: sqlite:///logs/my_model?timeout=60

``experiment_name``
    Groups all runs that share the same model or task under a single experiment in the UI.

``tracking_uri``
    Tells MLflow where to persist run data. Three formats are supported:

    * **Local file store** (default when omitted): data is written to an ``mlruns/`` directory
      next to the script.

      .. code:: yaml

          tracking_uri: mlruns

    * **SQLite** (used in all provided experiment configs): a single ``.db`` file avoids
      file-lock contention when running multiple processes in the same directory.

      .. code:: yaml

          tracking_uri: sqlite:///logs/my_model?timeout=60

    * **Remote server**: point to a running ``mlflow server`` instance for shared or
      cloud-based tracking.

      .. code:: yaml

          tracking_uri: http://my-mlflow-server:5000

.. note::
    The ``experiment`` optional dependency group is required to use MLflow.
    Install it with:

    .. parsed-literal::

        pip install torch-uncertainty[experiment]

Launching the MLflow UI
-----------------------

After running at least one experiment, start the UI with the command matching your
``tracking_uri``:

.. code:: bash

    # Local file store (mlruns/ in the current directory)
    mlflow ui

    # SQLite backend (adjust the path to match your tracking_uri)
    mlflow ui --backend-store-uri sqlite:///logs/my_model

    # Remote server (already running — just open the browser)

The UI is then available at ``http://localhost:5000``.

.. note::
    When using the provided Docker image, port ``5000`` is already exposed.
    No additional port-mapping flags are needed; run the command above inside
    the container and open ``http://localhost:5000`` on your host.

Navigating the UI
-----------------

Once the UI is open, you can:

* **Compare runs** side-by-side in the experiment table, filtering and sorting by any
  logged metric or parameter.
* **Plot metric curves** (training loss, validation accuracy, ECE, …) across steps or
  epochs for one or several runs at once.
* **Inspect logged artifacts**: reliability diagrams, OOD score histograms, risk–coverage
  curves, and the full ``config.yaml`` snapshot are attached to each run.
* **Reproduce a run** by downloading the saved ``config.yaml`` artifact and passing it
  back to the CLI.
