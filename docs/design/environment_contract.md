# Warehouse Environment Contract

## 1. Purpose

This document defines the interface between the existing
WarehouseEnv environment and the DDQN components that will
be developed by Member 2.

The environment is owned by Member 1.

The environment must remain independent of the DDQN
implementation.

---

## 2. Environment Configuration

The current environment is implemented in:

```text
src/environment/warehouse_env.py