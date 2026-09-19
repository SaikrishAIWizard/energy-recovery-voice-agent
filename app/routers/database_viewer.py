"""Unlisted, read-only database inspector for the local demo.

Hidden is NOT authenticated: keep the demo bound to loopback. Only the seven
explicitly allowlisted application tables are readable; no SQL, files, settings,
credentials, mutation routes, or relationship traversal are exposed.
"""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel
from sqlalchemy import String, cast, func, or_, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import (
    AuditEvent, CallSession, Handoff, JourneyField, JourneySubmission, Lead,
    TranscriptSegment,
)

router = APIRouter(prefix="/internal/database", include_in_schema=False)
TABLES = {
    model.__tablename__: model.__table__
    for model in (Lead, CallSession, JourneyField, TranscriptSegment,
                  Handoff, AuditEvent, JourneySubmission)
}


class ColumnInfo(BaseModel):
    name: str
    type: str
    nullable: bool
    primary_key: bool
    references: list[str]


class TableInfo(BaseModel):
    name: str
    row_count: int
    columns: list[ColumnInfo]


class TableIndex(BaseModel):
    read_only: Literal[True] = True
    tables: list[TableInfo]


class TableRows(BaseModel):
    table: str
    total: int
    matched: int
    limit: int
    offset: int
    sort_by: str
    direction: Literal["asc", "desc"]
    rows: list[dict[str, Any]]


def _no_store(response: Response) -> None:
    # The viewer can contain lead contact details and transcripts.
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Robots-Tag"] = "noindex, nofollow"


@router.get("/tables", response_model=TableIndex)
def list_tables(response: Response, db: Session = Depends(get_db)) -> TableIndex:
    _no_store(response)
    return TableIndex(tables=[
        TableInfo(
            name=name,
            row_count=db.execute(select(func.count()).select_from(table)).scalar_one(),
            columns=[ColumnInfo(
                name=column.name,
                type=str(column.type),
                nullable=column.nullable,
                primary_key=column.primary_key,
                references=sorted(key.target_fullname for key in column.foreign_keys),
            ) for column in table.columns],
        )
        for name, table in TABLES.items()
    ])


@router.get("/tables/{table_name}", response_model=TableRows)
def read_table(
    table_name: str,
    response: Response,
    limit: int = Query(25, ge=1, le=100),
    offset: int = Query(0, ge=0),
    q: str = Query("", max_length=200),
    sort_by: str | None = Query(None),
    direction: Literal["asc", "desc"] = "asc",
    db: Session = Depends(get_db),
) -> TableRows:
    table = TABLES.get(table_name)
    if table is None:
        raise HTTPException(status_code=404, detail="Unknown application table.")
    primary_keys = list(table.primary_key.columns)
    sort_name = sort_by if sort_by is not None else primary_keys[0].name
    if sort_name not in table.c:
        raise HTTPException(status_code=400, detail="Unknown sort column.")

    total = db.execute(select(func.count()).select_from(table)).scalar_one()
    statement = select(table)
    matched = total
    query = q.strip()
    if query:
        # SQLAlchemy binds values; wildcard characters are treated literally.
        predicate = or_(*(cast(column, String).contains(query, autoescape=True)
                          for column in table.columns))
        statement = statement.where(predicate)
        matched = db.execute(
            select(func.count()).select_from(table).where(predicate)
        ).scalar_one()

    column = table.c[sort_name]
    order = column.desc() if direction == "desc" else column.asc()
    # Primary keys break ties, so pagination stays stable for repeated values.
    statement = statement.order_by(
        order, *(key.asc() for key in primary_keys if key.name != sort_name)
    ).offset(offset).limit(limit)
    rows = [dict(row) for row in db.execute(statement).mappings()]
    _no_store(response)
    return TableRows(
        table=table_name, total=total, matched=matched, limit=limit, offset=offset,
        sort_by=sort_name, direction=direction, rows=rows,
    )
