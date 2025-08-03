"""
Pydantic models for database schema validation
Defines the structure for financial statements and company data with separate tables
"""

from pydantic import BaseModel, Field, validator
from datetime import datetime, date
from typing import Optional, Literal
from decimal import Decimal
import uuid


class Company(BaseModel):
    """Company information model"""
    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4)
    cik: str = Field(..., description="SEC Central Index Key")
    ticker: str = Field(..., description="Stock ticker symbol")
    name: str = Field(..., description="Company name")
    sic: Optional[str] = Field(None, description="Standard Industrial Classification")
    sic_description: Optional[str] = Field(None, description="SIC description")
    ein: Optional[str] = Field(None, description="Employer Identification Number")
    description: Optional[str] = Field(None, description="Company description")
    website: Optional[str] = Field(None, description="Company website")
    investor_website: Optional[str] = Field(None, description="Investor relations website")
    category: Optional[str] = Field(None, description="Company category")
    fiscal_year_end: Optional[str] = Field(None, description="Fiscal year end (MMDD format)")
    state_of_incorporation: Optional[str] = Field(None, description="State of incorporation")
    state_of_incorporation_description: Optional[str] = Field(None, description="State description")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('cik')
    def validate_cik(cls, v):
        if not v or len(v) != 10 or not v.isdigit():
            raise ValueError('CIK must be 10 digits')
        return v
    
    @validator('ticker')
    def validate_ticker(cls, v):
        if not v or len(v) > 10:
            raise ValueError('Ticker must be provided and max 10 characters')
        return v.upper()
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }


class BaseFinancialStatement(BaseModel):
    """Base class for all financial statements"""
    
    # Primary Key
    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4)
    
    # Company reference
    company_id: uuid.UUID = Field(..., description="Reference to company")
    cik: str = Field(..., description="Company CIK")
    ticker: str = Field(..., description="Company ticker")
    company_name: str = Field(..., description="Company name")
    
    # Period information
    period_end_date: date = Field(..., description="Period ending date")
    period_start_date: Optional[date] = Field(None, description="Period starting date")
    filing_date: Optional[date] = Field(None, description="SEC filing date")
    period_type: Literal["Q1", "Q2", "Q3", "Q4", "FY"] = Field(..., description="Quarter or full year")
    fiscal_year: int = Field(..., description="Fiscal year")
    period_length_months: int = Field(..., description="Period length in months", ge=1, le=12)
    form_type: Optional[str] = Field(None, description="SEC form type (10-Q, 10-K, etc.)")
    
    # Currency and units
    currency: str = Field(default="USD", description="Currency code")
    units: Literal["USD", "shares", "pure", "USD/shares"] = Field(default="USD", description="Unit type")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('period_length_months')
    def validate_period_length(cls, v, values):
        period_type = values.get('period_type')
        # Only validate FY periods to be 12 months, allow other periods to have varying lengths
        if period_type == 'FY' and v != 12:
            raise ValueError('Full year period must be 12 months')
        # For quarterly periods, allow standard quarterly lengths (3, 6, 9, 12)
        # But don't enforce strict validation as periods can vary in practice
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
            Decimal: lambda v: float(v) if v else None,
            uuid.UUID: lambda v: str(v)
        }


class IncomeStatement(BaseFinancialStatement):
    """Income Statement data model"""
    
    # Revenue
    total_revenue: Optional[Decimal] = Field(None, description="Total revenue")
    cost_of_revenue: Optional[Decimal] = Field(None, description="Cost of revenue")
    gross_profit: Optional[Decimal] = Field(None, description="Gross profit")
    
    # Operating Expenses
    research_and_development: Optional[Decimal] = Field(None, description="R&D expenses")
    sales_and_marketing: Optional[Decimal] = Field(None, description="Sales and marketing expenses")
    sales_general_and_admin: Optional[Decimal] = Field(None, description="Sales, general and administrative expenses")
    general_and_administrative: Optional[Decimal] = Field(None, description="General and administrative expenses")
    total_operating_expenses: Optional[Decimal] = Field(None, description="Total operating expenses")
    operating_income: Optional[Decimal] = Field(None, description="Operating income")
    
    # Non-Operating Items
    interest_income: Optional[Decimal] = Field(None, description="Interest income")
    interest_expense: Optional[Decimal] = Field(None, description="Interest expense")
    other_income: Optional[Decimal] = Field(None, description="Other non-operating income")
    income_before_taxes: Optional[Decimal] = Field(None, description="Income before taxes")
    income_tax_expense: Optional[Decimal] = Field(None, description="Income tax expense")
    net_income: Optional[Decimal] = Field(None, description="Net income")
    
    # Per Share Data
    earnings_per_share_basic: Optional[Decimal] = Field(None, description="Basic EPS")
    earnings_per_share_diluted: Optional[Decimal] = Field(None, description="Diluted EPS")
    weighted_average_shares_basic: Optional[Decimal] = Field(None, description="Weighted average shares basic")
    weighted_average_shares_diluted: Optional[Decimal] = Field(None, description="Weighted average shares diluted")


class BalanceSheet(BaseFinancialStatement):
    """Balance Sheet data model"""
    
    # Current Assets
    cash_and_cash_equivalents: Optional[Decimal] = Field(None, description="Cash and cash equivalents")
    short_term_investments: Optional[Decimal] = Field(None, description="Short-term investments")
    accounts_receivable: Optional[Decimal] = Field(None, description="Accounts receivable")
    inventory: Optional[Decimal] = Field(None, description="Inventory")
    prepaid_expenses: Optional[Decimal] = Field(None, description="Prepaid expenses")
    total_current_assets: Optional[Decimal] = Field(None, description="Total current assets")
    
    # Non-Current Assets
    property_plant_equipment: Optional[Decimal] = Field(None, description="Property, plant & equipment")
    goodwill: Optional[Decimal] = Field(None, description="Goodwill")
    intangible_assets: Optional[Decimal] = Field(None, description="Intangible assets")
    long_term_investments: Optional[Decimal] = Field(None, description="Long-term investments")
    other_assets: Optional[Decimal] = Field(None, description="Other assets")
    total_non_current_assets: Optional[Decimal] = Field(None, description="Total non-current assets")
    total_assets: Optional[Decimal] = Field(None, description="Total assets")
    
    # Current Liabilities
    accounts_payable: Optional[Decimal] = Field(None, description="Accounts payable")
    accrued_liabilities: Optional[Decimal] = Field(None, description="Accrued liabilities")
    commercial_paper: Optional[Decimal] = Field(None, description="Commercial paper")
    other_short_term_borrowings: Optional[Decimal] = Field(None, description="Other short-term borrowings")
    current_portion_long_term_debt: Optional[Decimal] = Field(None, description="Current portion of long-term debt")
    finance_lease_liability_current: Optional[Decimal] = Field(None, description="Current finance lease liability")
    operating_lease_liability_current: Optional[Decimal] = Field(None, description="Current operating lease liability")
    total_current_liabilities: Optional[Decimal] = Field(None, description="Total current liabilities")
    
    # Non-Current Liabilities
    long_term_debt: Optional[Decimal] = Field(None, description="Long-term debt")
    non_current_long_term_debt: Optional[Decimal] = Field(None, description="Non-current long-term debt")
    finance_lease_liability_noncurrent: Optional[Decimal] = Field(None, description="Non-current finance lease liability")
    operating_lease_liability_noncurrent: Optional[Decimal] = Field(None, description="Non-current operating lease liability")
    other_long_term_liabilities: Optional[Decimal] = Field(None, description="Other long-term liabilities")
    total_non_current_liabilities: Optional[Decimal] = Field(None, description="Total non-current liabilities")
    total_liabilities: Optional[Decimal] = Field(None, description="Total liabilities")
    
    # Equity
    common_stock: Optional[Decimal] = Field(None, description="Common stock")
    retained_earnings: Optional[Decimal] = Field(None, description="Retained earnings")
    accumulated_oci: Optional[Decimal] = Field(None, description="Accumulated other comprehensive income")
    total_stockholders_equity: Optional[Decimal] = Field(None, description="Total stockholders' equity")
    total_liabilities_and_equity: Optional[Decimal] = Field(None, description="Total liabilities and equity")


class CashFlowStatement(BaseFinancialStatement):
    """Cash Flow Statement data model"""
    
    # Operating Activities
    net_cash_from_operating_activities: Optional[Decimal] = Field(None, description="Net cash from operating activities")
    depreciation_and_amortization: Optional[Decimal] = Field(None, description="Depreciation and amortization")
    depreciation: Optional[Decimal] = Field(None, description="Depreciation")
    amortization: Optional[Decimal] = Field(None, description="Amortization")
    stock_based_compensation: Optional[Decimal] = Field(None, description="Stock-based compensation")
    changes_in_accounts_receivable: Optional[Decimal] = Field(None, description="Changes in accounts receivable")
    changes_in_other_receivable: Optional[Decimal] = Field(None, description="Changes in other receivables")
    changes_in_inventory: Optional[Decimal] = Field(None, description="Changes in inventory")
    changes_in_accounts_payable: Optional[Decimal] = Field(None, description="Changes in accounts payable")
    changes_in_other_operating_assets: Optional[Decimal] = Field(None, description="Changes in other operating assets")
    changes_in_other_operating_liabilities: Optional[Decimal] = Field(None, description="Changes in other operating liabilities")
    
    # Investing Activities
    net_cash_from_investing_activities: Optional[Decimal] = Field(None, description="Net cash from investing activities")
    capital_expenditures: Optional[Decimal] = Field(None, description="Capital expenditures")
    acquisitions: Optional[Decimal] = Field(None, description="Business acquisitions")
    purchases_of_intangible_assets: Optional[Decimal] = Field(None, description="Purchases of intangible assets")
    investments_purchased: Optional[Decimal] = Field(None, description="Investments purchased")
    investments_sold: Optional[Decimal] = Field(None, description="Investments sold")
    divestitures: Optional[Decimal] = Field(None, description="Divestitures")
    
    # Financing Activities
    net_cash_from_financing_activities: Optional[Decimal] = Field(None, description="Net cash from financing activities")
    dividends_paid: Optional[Decimal] = Field(None, description="Dividends paid")
    share_repurchases: Optional[Decimal] = Field(None, description="Share repurchases")
    proceeds_from_stock_issuance: Optional[Decimal] = Field(None, description="Proceeds from stock issuance")
    debt_issued: Optional[Decimal] = Field(None, description="Debt issued")
    debt_repaid: Optional[Decimal] = Field(None, description="Debt repaid")
    
    # Net Change in Cash
    net_change_in_cash: Optional[Decimal] = Field(None, description="Net change in cash")
    cash_beginning_of_period: Optional[Decimal] = Field(None, description="Cash at beginning of period")
    cash_end_of_period: Optional[Decimal] = Field(None, description="Cash at end of period")


class ProcessingMetadata(BaseModel):
    """Processing metadata for tracking data loading"""
    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4)
    ticker: str = Field(..., description="Company ticker")
    cik: str = Field(default="", description="Company CIK")
    processing_status: Literal["pending", "success", "partial", "error"] = Field(..., description="Processing status")
    periods_processed: int = Field(default=0, description="Number of periods successfully processed")
    periods_skipped: int = Field(default=0, description="Number of periods skipped")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    last_processed: datetime = Field(default_factory=datetime.utcnow, description="Last processing timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }
