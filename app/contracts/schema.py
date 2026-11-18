import json
from typing import Literal

from pydantic import BaseModel, Field, model_validator


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def qualified_name(schema: str, name: str) -> str:
    return f"{quote_identifier(schema)}.{quote_identifier(name)}"


def comment(value: str) -> str:
    return " ".join(value.splitlines())


class ValueCount(BaseModel):
    value: str
    count: int


class ColumnProfile(BaseModel):
    sample_rows: int
    null_count: int
    distinct_count: int | None = None
    top_values: list[ValueCount] = Field(default_factory=list)
    values_truncated: bool = False
    minimum: str | None = None
    maximum: str | None = None


class TableProfile(BaseModel):
    status: Literal["sampled", "empty", "skipped", "error"]
    row_limit: int = 1000
    sample_rows: int = 0
    message: str = ""


class ColumnDoc(BaseModel):
    name: str
    type: str
    description: str = ""
    nullable: bool = True
    is_pk: bool = False
    profile: ColumnProfile | None = None
    sample_values: list[str] = Field(default_factory=list)


class ForeignKeyDoc(BaseModel):
    name: str
    columns: list[str] = Field(min_length=1)
    referred_schema: str
    referred_table: str
    referred_columns: list[str] = Field(min_length=1)
    definition: str

    @model_validator(mode="after")
    def validate_column_pairs(self) -> "ForeignKeyDoc":
        if len(self.columns) != len(self.referred_columns):
            raise ValueError("Каждой колонке внешнего ключа должна соответствовать одна целевая колонка.")
        return self


class UniqueConstraintDoc(BaseModel):
    name: str
    columns: list[str] = Field(min_length=1)
    definition: str


class TableDoc(BaseModel):
    schema_name: str
    name: str
    kind: Literal["table", "view", "materialized_view"] = "table"
    description: str = ""
    columns: list[ColumnDoc] = Field(default_factory=list)
    primary_key: list[str] = Field(default_factory=list)
    foreign_keys: list[ForeignKeyDoc] = Field(default_factory=list)
    unique_constraints: list[UniqueConstraintDoc] = Field(default_factory=list)
    profile: TableProfile | None = None


class SchemaContext(BaseModel):
    source_id: str
    retrieval_mode: Literal["full_schema", "lexical", "pgvector", "hybrid"] = "full_schema"
    top_k_applied: bool = False
    tables: list[TableDoc] = Field(default_factory=list)
    join_paths: list[str] = Field(default_factory=list)

    def to_prompt(self) -> str:
        """Describe the selected catalog tables as SQL-like text for the SQL agent."""
        if self.top_k_applied:
            lines = [f"-- Отобранный фрагмент каталога схемы ({self.retrieval_mode})."]
        else:
            lines = [f"-- Полный каталог схемы ({self.retrieval_mode})."]
        for table in self.tables:
            name = qualified_name(table.schema_name, table.name)
            lines.append("")
            if table.description:
                lines.append(f"-- {comment(table.description)}")
            if table.kind != "table":
                lines.append(f"-- {table.kind}: {name}")
                for column in table.columns:
                    lines.append(f"-- {quote_identifier(column.name)} {column.type}: {comment(column.description)}")
                continue

            definitions = []
            for column in table.columns:
                definition = f"{quote_identifier(column.name)} {column.type}"
                if not column.nullable:
                    definition += " NOT NULL"
                definitions.append((definition, comment(column.description)))
            if table.primary_key:
                fields = ", ".join(map(quote_identifier, table.primary_key))
                definitions.append((f"PRIMARY KEY ({fields})", ""))
            for constraint in [*table.unique_constraints, *table.foreign_keys]:
                definitions.append((f"CONSTRAINT {quote_identifier(constraint.name)} {constraint.definition}", ""))

            lines.append(f"CREATE TABLE {name} (")
            for index, (definition, description) in enumerate(definitions):
                suffix = "," if index < len(definitions) - 1 else ""
                description_text = f" -- {description}" if description else ""
                lines.append(f"  {definition}{suffix}{description_text}")
            lines.append(");")

            for column in table.columns:
                if column.sample_values:
                    values = json.dumps(column.sample_values, ensure_ascii=False)
                    lines.append(f"-- Примеры {quote_identifier(column.name)} из ограниченной выборки, возможны другие значения: {values}")

        if self.join_paths:
            lines.append("\n-- Условия соединения из внешних ключей")
            lines.extend(f"-- JOIN {path}" for path in self.join_paths)
        return "\n".join(lines)
