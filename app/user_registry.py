async def reserve_user_id(conn, external_ref: str | None = None) -> int:
    row = await conn.fetchrow(
        "INSERT INTO everwear.user_id_registry(external_ref) "
        "VALUES ($1) RETURNING id",
        external_ref)
    return row["id"]
