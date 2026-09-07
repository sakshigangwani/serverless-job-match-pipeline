"""Shared data-contract package.

Deliberately has no eager imports: `common.models` depends on pydantic, but
`common.storage_keys` is pure stdlib. A Lambda that only needs hashing/key-building
(e.g. lambdas/fetch) should be able to `from common.storage_keys import ...` and be
bundled without pydantic. Import the submodule you need directly.
"""
