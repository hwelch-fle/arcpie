from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
from collections.abc import Iterator
from typing import Any, Literal, Self, TypedDict, get_args, overload
from builtins import range as py_range

from arcpy import da

from arcpy.management import (
    # Export/Import
    DomainToTable, # type: ignore
    TableToDomain, # type: ignore
    
    # CRUD
    CreateDomain, # type: ignore
    AlterDomain, # type: ignore
    DeleteDomain, # type: ignore
    
    # Assignment
    AssignDomainToField, # type: ignore
    RemoveDomainFromField, # type: ignore
    
    # Range
    SetValueForRangeDomain, # type: ignore
    
    # CodedValue
    SortCodedValueDomain, # type: ignore
    AddCodedValueToDomain, # type: ignore
    DeleteCodedValueFromDomain, # type: ignore
)

from ..featureclass import FeatureClass, Table

TYPE_CHECKING = False
if TYPE_CHECKING:
    from ..database import Dataset
    from .field import FieldType


__all__ = ('Domain', 'CodedValueDomain', 'RangeDomain')


type NumericType = int | float
type DateType = datetime | date | time
type ValueType = str | NumericType | DateType
type Description = str
type CodedValues = dict[ValueType, Description]
type RangeValue = tuple[ValueType, ValueType] | py_range

# Used for CodedValueDomain.codedValues.setter
# None signals code deletion
type CodedValuesNullable = dict[ValueType, Description | None]


# Argument maps given by Domain attribute -> expected by Domain funcs

def map_args(fr: Any, to: Any) -> dict[Any, Any]:
    """Map two literals"""
    return dict(zip(get_args(fr), get_args(to)))


# Domain.type   
DomainFieldType  = Literal['Short', 'Long', 'BigInteger', 'Float', 'Double', 'Text', 'Date', 'DateOnly', 'TimeOnly']
_DomainFieldType = Literal['SHORT', 'LONG', 'BIGINTEGER', 'FLOAT', 'DOUBLE', 'TEXT', 'DATE', 'DATEONLY', 'TIMEONLY']
_DOMAIN_FIELD_TYPE_MAP: dict[DomainFieldType, _DomainFieldType] = map_args(DomainFieldType, _DomainFieldType)


# Domain.mergePolicy
MergePolicy  = Literal['DefaultValue', 'AreaWeighted', 'SumValues']
_MergePolicy = Literal['DEFAULT', 'AREA_WEIGHTED', 'SUM_VALUES']
_DOMAIN_MERGE_POLICY_MAP: dict[MergePolicy, _MergePolicy] = map_args(MergePolicy, _MergePolicy)


# Domain.splitPolicy
SplitPolicy  = Literal['DefaultValue', 'Duplicate', 'GeometryRatio']
_SplitPolicy = Literal['DEFAULT', 'DUPLICATE', 'GEOMETRY_RATIO']
_DOMAIN_SPLIT_POLICY_MAP: dict[SplitPolicy, _SplitPolicy] = map_args(SplitPolicy, _SplitPolicy)


# Domain.domainType
DomainType  = Literal['CodedValue', 'Range']
_DomainType = Literal['CODED', 'RANGE']
_DOMAIN_TYPE_MAP: dict[DomainType, _DomainType] = map_args(DomainType, _DomainType)


# Map Domain Types to valid Field types
VALID_FIELD_TYPES_FOR: dict[DomainFieldType, set[FieldType]] = {
    'Short': {'SHORT'}, 
    'Long': {'LONG', 'SHORT'}, 
    'BigInteger': {'BIGINTEGER', 'LONG', 'SHORT'}, 
    'Float': {'FLOAT'}, 
    'Double': {'DOUBLE', 'FLOAT'}, 
    'Text': {'TEXT'}, 
    'TimeOnly': {'TIMEONLY'},
    'DateOnly': {'DATEONLY'}, 
    'Date': {'DATE', 'DATEHIGHPRECISION', 'DATEONLY', 'TIMEONLY'}, 
}


class DomainSchema(TypedDict):
    name: str
    domainType: DomainType
    type: DomainFieldType
    description: Description
    codedValues: CodedValues | None
    range: RangeValue | None
    mergePolicy: MergePolicy
    splitPolicy: SplitPolicy
    owner: str


class BaseDomain:
    """Base class for interacting with Domains.
    
    General setup structure is pulled from pathlib.Path since 
    """
    def __new__(cls, wrapped: da.Domain, *args: Any, **kwargs: Any):
        if cls is Domain:
            _dtype = wrapped.domainType
            match wrapped.domainType:
                case 'CodedValue':
                    cls = CodedValueDomain
                case 'Range':
                    cls = RangeDomain
                case _:
                    raise TypeError(f'Domain of type {_dtype} is not supported')
        return object.__new__(cls)
        
    def __init__(self, wrapped: da.Domain, workspace: str | Path | Dataset) -> None:
        from ..database import Dataset
        self._domain = wrapped
        if not isinstance(workspace, Dataset):
            self.dataset = Dataset(workspace)
            self.workspace = str(workspace)
        else:
            self.dataset = workspace
            self.workspace = str(workspace)

    def sync(self) -> None:
        """Sync the domain with the workspace"""
        matches = [d for d in da.ListDomains(self.workspace) if d.name == self.name]
        if not matches:
            raise LookupError(f'Domain {self.name} no longer exists in the parent workspace {self.workspace}')
        self._domain = matches.pop()

    # read only
    @property
    def type(self) -> da.DomainFieldType:
        return self._domain.type
    
    @property
    def domainType(self) -> DomainType:
        return self._domain.domainType
    
    # read/write
    @property
    def description(self) -> str:
        return self._domain.description
    
    @description.setter
    def description(self, description: Description) -> None:
        AlterDomain(self.workspace, self.name, new_domain_description=description)
        self.sync()
    
    @property
    def mergePolicy(self) -> MergePolicy:
        return self._domain.mergePolicy
    
    @mergePolicy.setter
    def mergePolicy(self, merge_policy: MergePolicy) -> None:
        flag_map = _DOMAIN_MERGE_POLICY_MAP
        if merge_policy not in flag_map:
            raise ValueError(f'{merge_policy} is not one of {list(flag_map.keys())}')
        AlterDomain(self.workspace, self.name, merge_policy=flag_map[merge_policy])
        self.sync()
    
    @property
    def splitPolicy(self) -> SplitPolicy:
        return self._domain.splitPolicy
    
    @splitPolicy.setter
    def splitPolicy(self, split_policy: SplitPolicy) -> None:
        flag_map = _DOMAIN_SPLIT_POLICY_MAP
        if split_policy not in flag_map:
            raise ValueError(f'{split_policy} is not one of {list(flag_map.keys())}')
        AlterDomain(self.workspace, self.name, split_policy=flag_map[split_policy])
        self.sync()
        
    @property
    def name(self) -> str:
        return self._domain.name
    
    @name.setter
    def name(self, name: str) -> None:
        AlterDomain(self.workspace, self.name, new_domain_name=name)
        self.sync()
        
    @property
    def owner(self) -> str:
        return self._domain.owner
    
    @owner.setter
    def owner(self, owner: str) -> None:
        AlterDomain(self.workspace, self.name, new_domain_owner=owner)
        self.sync()


type FieldUsage = dict[str, list[str]]


class Domain(BaseDomain):
    """General Domain object"""

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.name})'

    def to_dict(self) -> DomainSchema:
        """Use with json.dump/dumps"""
        return {
            'name': self.name,
            'domainType': self.domainType,
            'type': self.type,
            'description': self.description,
            'codedValues': self.codedValues,
            'range': self.range,
            'mergePolicy': self.mergePolicy,
            'splitPolicy': self.splitPolicy,
            'owner': self.owner,
        }

    def used_by(self) -> FieldUsage:
        """Get a mapping of FeatureClass/Table -> [Field, ...] usage for the domain"""
        return self.dataset.domains.usage(self.name)[self.name]

    def assign_to(self, table: Table | FeatureClass, field: str,
                 *,
                 subtype_code: int | None = None,
                 ) -> None:
        """Assign the domain to a Field
        
        Args:
            table: The table to apply the domain to
            field: The field in the table to apply the domain to
            subtype_code: An optional subtype to apply the domain to
        """
        field_info = table.field_defs.get(field)
        if not field_info:
            raise ValueError(f'{table.name}: {field} not found!')
        if field_info.get('field_type') not in VALID_FIELD_TYPES_FOR.get(self.type, set()):
            msg = (
                f'Domain of type {self.type} cannot be assigned to '
                f'field {field} of type {field_info.get('field_type')}'
            )
            raise ValueError(msg)

        AssignDomainToField(
            in_table=str(table),
            field_name=field,
            domain_name=self.name,
            subtype_code=subtype_code,
        )

    def delete(self) -> None:
        """Delete the domain"""
        DeleteDomain(self.workspace, self.name)
    
    def add_to(self, workspace: Dataset) -> None:
        """Add the domain to a root workspace/Dataset"""
        if workspace.parent:
            raise ValueError(f'Domains can only be added to the root Dataset!')
        CreateDomain(
            in_workspace=workspace, 
            domain_name=self.name, 
            domain_description=self.description,
            field_type=_DOMAIN_FIELD_TYPE_MAP[self.type],
            domain_type=_DOMAIN_TYPE_MAP[self.domainType],
            split_policy=_DOMAIN_SPLIT_POLICY_MAP[self.splitPolicy],
            merge_policy=_DOMAIN_MERGE_POLICY_MAP[self.mergePolicy],
        )
    
    def to_table(self, table_name: str, 
                 *,
                 code_field: str = 'code',
                 description_field: str = 'description',
                 configuration_keyword: str | None = None
                 ) -> Table:
        """Convert the domain to a Table in its workspace
        
        Args:
            table_name: The name of the output table
            code_field: The field name for the codes
            description_field: The field name for the descriptions
            configuration_keyword: An optional config keyword for the database
        """
        if self.domainType != 'CodedValue':
            raise TypeError(f'Only `CodedValue` domains can be converted to a table')
        return Table(
            *DomainToTable(
                in_workspace=self.workspace, 
                domain_name=self.name, 
                out_table=table_name,
                code_field=code_field,
                description_field=description_field,
                configuration_keyword=configuration_keyword,
            )
        )
    
    
    # exposed props from subclasses
    # isinstance will typenarrow. 
    # Using base Domain constructor will always create the correct 
    # subclass at runtime.
    
    # Range props
    @property 
    def range(self) -> tuple[ValueType, ValueType] | None:
        return self._domain.range
    
    # CodedValue props
    @property
    def codedValues(self) -> CodedValues | None:
        return self._domain.codedValues
    
    def sort(self, by: Literal['code', 'value'], order: Literal['asc', 'desc']) -> None:
        """CodedValue Only"""
    
    # constructors
    @overload
    @classmethod
    def create(cls, workspace: str, name: str, 
               *,
               domain_type: Literal['CodedValue'],
               field_type: DomainFieldType,
               coded_values: CodedValues,
               description: str, 
               split_policy: SplitPolicy = ...,
               merge_policy: MergePolicy = ...,
               ) -> CodedValueDomain: ...
    
    @overload
    @classmethod
    def create(cls, workspace: str, name: str, 
               *,
               domain_type: Literal['Range'],
               field_type: DomainFieldType,
               domain_range: RangeValue,
               description: str,
               split_policy: SplitPolicy = ...,
               merge_policy: MergePolicy = ...,
               ) -> RangeDomain: ...
    
    @classmethod
    def create(cls, workspace: str, name: str, 
               *,
               domain_type: DomainType,
               field_type: DomainFieldType,
               description: str = '', 
               split_policy: SplitPolicy = 'Duplicate',
               merge_policy: MergePolicy = 'DefaultValue',
               
               # Required by overload
               domain_range: RangeValue | None = None,
               coded_values: CodedValues | None = None) -> CodedValueDomain | RangeDomain:
        """Create a new domain"""
        assert domain_type in ['CodedValue', 'Range'], f'Invalid domain type {domain_type}'
        
        CreateDomain(
            in_workspace=workspace,
            domain_name=name,
            domain_description=description,
            field_type=_DOMAIN_FIELD_TYPE_MAP.get(field_type),
            domain_type=_DOMAIN_TYPE_MAP.get(domain_type),
            split_policy=_DOMAIN_SPLIT_POLICY_MAP.get(split_policy),
            merge_policy=_DOMAIN_MERGE_POLICY_MAP.get(merge_policy),
        )
        new_domain, *_ = [d for d in da.ListDomains(workspace) if d.name == name] or [None]
        if new_domain is None:
            raise ValueError(f'Something went wrong creating the Domain...')
        if domain_type == 'CodedValue':
            domain = CodedValueDomain(new_domain, workspace)
            if coded_values:
                domain.codedValues = {k: v for k, v in coded_values.items()}
            return domain
        elif domain_type == 'Range':
            domain = RangeDomain(new_domain, workspace)
            if domain_range:
                domain.range = domain_range
            return domain

    @classmethod
    def from_table(cls, table: FeatureClass | Table, name: str,
                   *,
                   code_field: str,
                   description_field: str,
                   workspace: str | None = None,
                   description: str | None,
                   overwrite_existing: bool = False,
                   ) -> CodedValueDomain:
        """Create a CodedValue domain from a Table
        
        Args:
            table: The table or featureclass to create the domain from
            name: The name of the domain
            code_field: The field to use for codes
            description_field: The field to use for descriptions
            workspace: The workspace to create the domain in (default: table.workspace)
            description: An optional description for the new domain
            overwrite_existing: If set to `True`, any existing domains with the name will be replaced,<br> 
                otherwise new values are appended to the existing domain
        """
        if code_field not in table.fields or description_field not in table.fields:
            msg = (
                f'Invalid {code_field} or {description_field} not in table fields, '
                f'must be one of {table.fields}'
            )
            raise ValueError(msg)
        
        workspace = workspace or table.workspace
        TableToDomain(
            in_table=str(table),
            code_field=code_field,
            description_field=description_field,
            in_workspace=workspace or table.workspace,
            domain_name=name,
            domain_description=description,
            update_option='REPLACE' if overwrite_existing else 'APPEND'
        )
        return CodedValueDomain([d for d in da.ListDomains(workspace) if d.name == name].pop(), workspace)


class RangeDomain(Domain):
    """Domain with a Range component"""
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.name}, {self.range})'
    
    if TYPE_CHECKING:
        # Narrow type for __new__
        def __new__(cls, wrapped: da.Domain, workspace: str | Path | Dataset[Any]) -> Self: ...
    
    @property
    def range(self) -> tuple[ValueType, ValueType]:
        return self._domain.range
    
    @range.setter
    def range(self, domain_range: RangeValue) -> None:
        SetValueForRangeDomain(self.workspace, self.name, domain_range[0], domain_range[1])
        self.sync()

    @property
    def domainType(self) -> Literal['Range']:
        return 'Range'


class CodedValueDomain(Domain):
    """Domain with a CodedValue component"""
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.name}, {self.codedValues})'
    
    if TYPE_CHECKING:
        # Narrow type for __new__
        def __new__(cls, wrapped: da.Domain, workspace: str | Path | Dataset[Any]) -> Self: ...
    
    @property
    def codedValues(self) -> CodedValues:
        return self._domain.codedValues
    
    @codedValues.setter
    def codedValues(self, update_values: CodedValuesNullable) -> None:
        """Update the coded values for the domain. 
        
        Note: 
        Values will be used to *update* the current values, to remove, set description to `None` 
        If you need an empty description, set the description to an empty string
        
        Example:
            ```python
            >>> dom = CodedValueDomain(...)
            >>> dom.codedValues
            {'a': 'first', 'b': 'second'}
            >>> dom.codedValues = {'c': 'third'}
            >>> dom.codedValues
            {'a': 'first', 'b': 'second', 'c': 'third'}
            >>> dom.codedValues = {'a': None, 'b': 'first'}
            >>> dom.codedValues
            {'b': 'first', 'c': 'third'}
            ```
        """
        codes: CodedValuesNullable = {**self.codedValues, **update_values}
        for code, description in codes.items():
            # Delete existing code
            if code in self.codedValues:
                DeleteCodedValueFromDomain(self.workspace, self.name, code)
            # Add the code if the description is not flagged for deletion
            if description is not None:
                AddCodedValueToDomain(self.workspace, self.name, code, description)
        self.sync()
        
    @property
    def domainType(self) -> Literal['CodedValue']:
        return 'CodedValue'
    
    def sort(self, by: Literal['code', 'value'], order: Literal['asc', 'desc']) -> None:
        SortCodedValueDomain(
            self.workspace,
            domain_name=self.name,
            sort_by='CODE' if by == 'code' else 'DESCRIPTION',
            sort_order='ASCENDING' if order == 'asc' else 'DESCENDING',
        )
    

type DomainUsageMap = dict[str, FieldUsage]


class DomainManager:
    """Container for managing dataset domains"""
    
    def __init__(self, dataset: Dataset[Any]) -> None:
        self.dataset = dataset
        self.workspace = str(dataset)
    
    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}'
            f'({self.dataset.name}, '
            f'RangeDomains: {len(self.range_domains)} , '
            f'CodedValueDomains: {len(self.coded_value_domains)})'
        )
    
    @property
    def domains(self) -> list[Domain]:
        return [
            Domain(d, self.dataset)
            for d in da.ListDomains(self.workspace)
        ]
    
    @property
    def coded_value_domains(self) -> list[CodedValueDomain]:
        return [
            d for d in self.domains
            if isinstance(d, CodedValueDomain)
        ]
    
    @property
    def range_domains(self) -> list[RangeDomain]:
        return [
            d for d in self.domains
            if isinstance(d, RangeDomain)
        ]
    
    @property
    def domain_names(self) -> list[str]:
        return [d.name for d in self.domains]
    
    def __iter__(self) -> Iterator[Domain]:
        return iter(self.domains)
    
    def __len__(self) -> int:
        return len(self.domains)
    
    def __contains__(self, domain: str) -> bool:
        return domain in self.domain_names
    
    def __getitem__(self, domain: str) -> Domain:
        for d in self.domains:
            if d.name == domain:
                return d
        raise KeyError(f'{domain} is not a domain in {self.dataset.name}')
    
    def get[D](self, domain: str, default: D = None) -> Domain | D:
        try:
            return self[domain]
        except KeyError:
            return default
    
    def usage(self, *domain_names: str) -> DomainUsageMap:
        """A mapping of domains to features to fields that shows usage of a domain in a dataset
        
        Args:
            *domain_names: Varargs of all domain names to include in the output mapping
        
        Returns:
            A Nested mapping of `Domain Name -> Feature Class -> [Field Name, ...]`
        """
        
        if not domain_names:
            domain_names = tuple(self.domain_names)
        
        schema = self.dataset.schema
        fc_usage: DomainUsageMap = {}
        for dataset in schema['datasets']:
            features = dataset.get('datasets') or [dataset]
            for feature in features:
                if 'fields' not in feature:
                    continue
                for field in feature['fields']['fieldArray']:
                    if 'domain' not in field:
                        continue
                    field_domain = field['domain']['domainName']
                    fc_usage.setdefault(field_domain, {})
                    if field_domain in domain_names:
                        fc_usage[field_domain].setdefault(feature['name'], [])
                        fc_usage[field_domain][feature['name']].append(field.get('name', '...'))
        return fc_usage
    
    def to_dict(self) -> dict[str, DomainSchema]:
        """Use with json.dump/dumps"""
        return {d.name: d.to_dict() for d in self.domains}
        