"""
PostgreSQL database connection and operations for three-table structure
Handles table creation, data insertion, and duplicate detection for separate statement tables
"""

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, execute_values
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    psycopg2 = None
    RealDictCursor = None
    execute_values = None

import os
from typing import List, Optional, Dict, Any, Tuple, Union
import logging
from datetime import datetime, date
import uuid
from decimal import Decimal
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from .models import Company, IncomeStatement, BalanceSheet, CashFlowStatement, ProcessingMetadata
except ImportError:
    from models import Company, IncomeStatement, BalanceSheet, CashFlowStatement, ProcessingMetadata

logger = logging.getLogger(__name__)


class FinancialDatabase:
    """PostgreSQL database handler for financial statements with separate tables"""
    
    # Whitelist of allowed table names for SQL injection prevention
    ALLOWED_TABLES = {
        'companies',
        'income_statements', 
        'balance_sheets',
        'cash_flow_statements',
        'ratio_definitions',
        'calculated_ratios',
        'ltm_income_statements',
        'ltm_cash_flow_statements'
    }
    
    def __init__(self):
        if not PSYCOPG2_AVAILABLE:
            raise ImportError(
                "psycopg2 is required for database operations. "
                "Install it with: pip install psycopg2-binary"
            )
        
        self.connection_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5433')),
            'database': os.getenv('DB_NAME', 'financial_data'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', ''),
        }
        
        self.connection = None
        self._ensure_connection()
        self._create_tables()
    
    def _ensure_connection(self):
        """Ensure database connection is active"""
        try:
            if self.connection is None or self.connection.closed:
                self.connection = psycopg2.connect(**self.connection_params)
                self.connection.autocommit = False
                logger.info("Database connection established")
        except psycopg2.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def _validate_table_name(self, table_name: str) -> str:
        """
        Validate table name against whitelist to prevent SQL injection
        
        Args:
            table_name: Table name to validate
            
        Returns:
            Validated table name
            
        Raises:
            ValueError: If table name is not in whitelist
        """
        if table_name not in self.ALLOWED_TABLES:
            raise ValueError(f"Invalid table name: {table_name}. Allowed tables: {', '.join(self.ALLOWED_TABLES)}")
        return table_name
    
    def _sanitize_ticker(self, ticker: str) -> str:
        """
        Sanitize ticker input to prevent SQL injection
        
        Args:
            ticker: Ticker symbol to sanitize
            
        Returns:
            Sanitized ticker symbol
            
        Raises:
            ValueError: If ticker contains invalid characters
        """
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Ticker must be a non-empty string")
        
        # Remove any potential SQL injection characters
        sanitized = ticker.strip().upper()
        
        # Allow only alphanumeric characters, dots, and hyphens
        import re
        if not re.match(r'^[A-Z0-9.-]+$', sanitized):
            raise ValueError(f"Invalid ticker format: {ticker}. Only alphanumeric characters, dots, and hyphens allowed")
        
        if len(sanitized) > 10:
            raise ValueError(f"Ticker too long: {ticker}. Maximum 10 characters allowed")
        
        return sanitized
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        try:
            with self.connection.cursor() as cursor:
                # Companies table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS companies (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        cik VARCHAR(10) UNIQUE NOT NULL,
                        ticker VARCHAR(10) NOT NULL,
                        name TEXT NOT NULL,
                        sic VARCHAR(10),
                        sic_description TEXT,
                        ein VARCHAR(20),
                        description TEXT,
                        website TEXT,
                        investor_website TEXT,
                        category TEXT,
                        fiscal_year_end VARCHAR(4),
                        state_of_incorporation VARCHAR(10),
                        state_of_incorporation_description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Income Statements table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS income_statements (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        company_id UUID NOT NULL REFERENCES companies(id),
                        cik VARCHAR(10) NOT NULL,
                        ticker VARCHAR(10) NOT NULL,
                        company_name TEXT NOT NULL,
                        
                        -- Period information
                        period_end_date DATE NOT NULL,
                        period_start_date DATE,
                        filing_date DATE,
                        period_type VARCHAR(2) NOT NULL CHECK (period_type IN ('Q1', 'Q2', 'Q3', 'Q4', 'FY')),
                        fiscal_year INTEGER NOT NULL,
                        period_length_months INTEGER NOT NULL CHECK (period_length_months IN (3, 6, 9, 12)),
                        form_type VARCHAR(10),
                        
                        -- Currency and units
                        currency VARCHAR(3) DEFAULT 'USD',
                        units VARCHAR(10) DEFAULT 'USD',
                        
                        -- Income Statement fields
                        total_revenue DECIMAL(20,2),
                        cost_of_revenue DECIMAL(20,2),
                        gross_profit DECIMAL(20,2),
                        research_and_development DECIMAL(20,2),
                        sales_and_marketing DECIMAL(20,2),
                        sales_general_and_admin DECIMAL(20,2),
                        general_and_administrative DECIMAL(20,2),
                        total_operating_expenses DECIMAL(20,2),
                        operating_income DECIMAL(20,2),
                        ebitda DECIMAL(20,2),
                        interest_income DECIMAL(20,2),
                        interest_expense DECIMAL(20,2),
                        other_income DECIMAL(20,2),
                        income_before_taxes DECIMAL(20,2),
                        income_tax_expense DECIMAL(20,2),
                        net_income DECIMAL(20,2),
                        earnings_per_share_basic DECIMAL(10,4),
                        earnings_per_share_diluted DECIMAL(10,4),
                        weighted_average_shares_basic DECIMAL(20,0),
                        weighted_average_shares_diluted DECIMAL(20,0),
                        
                        -- Metadata
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        
                        -- Constraints
                        UNIQUE(cik, period_end_date, period_type, period_length_months)
                    );
                """)
                
                # Balance Sheets table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS balance_sheets (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        company_id UUID NOT NULL REFERENCES companies(id),
                        cik VARCHAR(10) NOT NULL,
                        ticker VARCHAR(10) NOT NULL,
                        company_name TEXT NOT NULL,
                        
                        -- Period information
                        period_end_date DATE NOT NULL,
                        period_start_date DATE,
                        filing_date DATE,
                        period_type VARCHAR(2) NOT NULL CHECK (period_type IN ('Q1', 'Q2', 'Q3', 'Q4', 'FY')),
                        fiscal_year INTEGER NOT NULL,
                        period_length_months INTEGER NOT NULL CHECK (period_length_months IN (3, 6, 9, 12)),
                        form_type VARCHAR(10),
                        
                        -- Currency and units
                        currency VARCHAR(3) DEFAULT 'USD',
                        units VARCHAR(10) DEFAULT 'USD',
                        
                        -- Balance Sheet fields
                        cash_and_cash_equivalents DECIMAL(20,2),
                        short_term_investments DECIMAL(20,2),
                        cash_and_short_term_investments DECIMAL(20,2), -- Combined cash + short-term investments
                        accounts_receivable DECIMAL(20,2),
                        inventory DECIMAL(20,2),
                        prepaid_expenses DECIMAL(20,2),
                        total_current_assets DECIMAL(20,2),
                        property_plant_equipment DECIMAL(20,2),
                        goodwill DECIMAL(20,2),
                        intangible_assets DECIMAL(20,2),
                        long_term_investments DECIMAL(20,2),
                        other_assets DECIMAL(20,2),
                        total_non_current_assets DECIMAL(20,2),
                        total_assets DECIMAL(20,2),
                        accounts_payable DECIMAL(20,2),
                        accrued_liabilities DECIMAL(20,2),
                        commercial_paper DECIMAL(20,2),
                        other_short_term_borrowings DECIMAL(20,2),
                        current_portion_long_term_debt DECIMAL(20,2),
                        finance_lease_liability_current DECIMAL(20,2),
                        operating_lease_liability_current DECIMAL(20,2),
                        total_current_liabilities DECIMAL(20,2),
                        long_term_debt DECIMAL(20,2),
                        non_current_long_term_debt DECIMAL(20,2),
                        finance_lease_liability_noncurrent DECIMAL(20,2),
                        operating_lease_liability_noncurrent DECIMAL(20,2),
                        other_long_term_liabilities DECIMAL(20,2),
                        total_non_current_liabilities DECIMAL(20,2),
                        total_liabilities DECIMAL(20,2),
                        common_stock DECIMAL(20,2),
                        retained_earnings DECIMAL(20,2),
                        accumulated_oci DECIMAL(20,2),
                        total_stockholders_equity DECIMAL(20,2),
                        total_liabilities_and_equity DECIMAL(20,2),
                        
                        -- Metadata
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        
                        -- Constraints
                        UNIQUE(cik, period_end_date, period_type, period_length_months)
                    );
                """)
                
                # Cash Flow Statements table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cash_flow_statements (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        company_id UUID NOT NULL REFERENCES companies(id),
                        cik VARCHAR(10) NOT NULL,
                        ticker VARCHAR(10) NOT NULL,
                        company_name TEXT NOT NULL,
                        
                        -- Period information
                        period_end_date DATE NOT NULL,
                        period_start_date DATE,
                        filing_date DATE,
                        period_type VARCHAR(2) NOT NULL CHECK (period_type IN ('Q1', 'Q2', 'Q3', 'Q4', 'FY')),
                        fiscal_year INTEGER NOT NULL,
                        period_length_months INTEGER NOT NULL CHECK (period_length_months IN (3, 6, 9, 12)),
                        form_type VARCHAR(10),
                        
                        -- Currency and units
                        currency VARCHAR(3) DEFAULT 'USD',
                        units VARCHAR(10) DEFAULT 'USD',
                        
                        -- Cash Flow Statement fields
                        net_cash_from_operating_activities DECIMAL(20,2),
                        depreciation_and_amortization DECIMAL(20,2),
                        depreciation DECIMAL(20,2),
                        amortization DECIMAL(20,2),
                        stock_based_compensation DECIMAL(20,2),
                        changes_in_accounts_receivable DECIMAL(20,2),
                        changes_in_other_receivable DECIMAL(20,2),
                        changes_in_inventory DECIMAL(20,2),
                        changes_in_accounts_payable DECIMAL(20,2),
                        changes_in_other_operating_assets DECIMAL(20,2),
                        changes_in_other_operating_liabilities DECIMAL(20,2),
                        net_cash_from_investing_activities DECIMAL(20,2),
                        capital_expenditures DECIMAL(20,2),
                        acquisitions DECIMAL(20,2),
                        purchases_of_intangible_assets DECIMAL(20,2),
                        investments_purchased DECIMAL(20,2),
                        investments_sold DECIMAL(20,2),
                        divestitures DECIMAL(20,2),
                        net_cash_from_financing_activities DECIMAL(20,2),
                        dividends_paid DECIMAL(20,2),
                        share_repurchases DECIMAL(20,2),
                        proceeds_from_stock_issuance DECIMAL(20,2),
                        debt_issued DECIMAL(20,2),
                        debt_repaid DECIMAL(20,2),
                        net_change_in_cash DECIMAL(20,2),
                        cash_beginning_of_period DECIMAL(20,2),
                        cash_end_of_period DECIMAL(20,2),
                        
                        -- Metadata
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        
                        -- Constraints
                        UNIQUE(cik, period_end_date, period_type, period_length_months)
                    );
                """)
                
                # Processing metadata table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS processing_metadata (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        ticker VARCHAR(10) NOT NULL,
                        cik VARCHAR(10) NOT NULL,
                        processing_status VARCHAR(20) CHECK (processing_status IN ('pending', 'success', 'partial', 'error')),
                        periods_processed INTEGER DEFAULT 0,
                        periods_skipped INTEGER DEFAULT 0,
                        error_message TEXT,
                        last_processed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(ticker, cik)
                    );
                """)
                
                # Ratio definitions table (hybrid: global and company-specific)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ratio_definitions (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        name VARCHAR(100) NOT NULL,
                        company_id UUID REFERENCES companies(id), -- NULL = global ratio
                        formula TEXT NOT NULL,
                        description TEXT,
                        category VARCHAR(50),
                        is_active BOOLEAN DEFAULT true,
                        created_by VARCHAR(100),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(name, company_id) -- Allows same name for global and company-specific
                    );
                """)
                
                # Calculated ratios table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS calculated_ratios (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        company_id UUID NOT NULL REFERENCES companies(id),
                        ratio_definition_id UUID NOT NULL REFERENCES ratio_definitions(id),
                        ticker VARCHAR(10) NOT NULL,
                        ratio_name VARCHAR(100) NOT NULL,  -- Add ratio name for easy querying
                        ratio_category VARCHAR(50),  -- Add ratio category from ratio definitions
                        period_end_date DATE NOT NULL,
                        period_type VARCHAR(3) NOT NULL CHECK (period_type IN ('Q1', 'Q2', 'Q3', 'Q4', 'FY', 'LTM')),
                        fiscal_year INTEGER NOT NULL,  -- Make fiscal_year NOT NULL
                        fiscal_quarter VARCHAR(3),  -- Add fiscal quarter (Q1, Q2, Q3, Q4)
                        ratio_value DECIMAL(15,6),
                        calculation_inputs JSONB, -- Store the actual values used in calculation
                        data_source VARCHAR(20) DEFAULT 'LTM', -- 'LTM', 'quarterly', 'annual'
                        calculation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(company_id, ratio_definition_id, period_end_date, period_type, data_source)
                    );
                """)
                
                # LTM Income Statements table - mirrors income_statements structure but stores 12-month rolling calculations
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ltm_income_statements (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        company_id UUID NOT NULL REFERENCES companies(id),
                        cik VARCHAR(10) NOT NULL,
                        ticker VARCHAR(10) NOT NULL,
                        company_name TEXT NOT NULL,
                        
                        -- LTM Period information  
                        period_end_date DATE NOT NULL,  -- End date of the 12-month period (standardized field name)
                        ltm_period_start_date DATE,         -- Start date of the 12-month period (period_end_date - 365 days)
                        base_quarter_end_date DATE NOT NULL, -- The quarter end date this LTM calculation is based on
                        fiscal_year INTEGER NOT NULL,
                        period_type VARCHAR(3) NOT NULL,  -- Q1, Q2, Q3, Q4 (quarter that ends the LTM period) - standardized field name
                        form_type VARCHAR(10),
                        
                        -- Currency and units
                        currency VARCHAR(3) DEFAULT 'USD',
                        units VARCHAR(10) DEFAULT 'USD',
                        
                        -- LTM Income Statement fields (same as income_statements table)
                        total_revenue DECIMAL(20,2),
                        cost_of_revenue DECIMAL(20,2),
                        gross_profit DECIMAL(20,2),
                        research_and_development DECIMAL(20,2),
                        sales_and_marketing DECIMAL(20,2),
                        sales_general_and_admin DECIMAL(20,2),
                        general_and_administrative DECIMAL(20,2),
                        total_operating_expenses DECIMAL(20,2),
                        operating_income DECIMAL(20,2),
                        ebitda DECIMAL(20,2),
                        interest_income DECIMAL(20,2),
                        interest_expense DECIMAL(20,2),
                        other_income DECIMAL(20,2),
                        income_before_taxes DECIMAL(20,2),
                        income_tax_expense DECIMAL(20,2),
                        net_income DECIMAL(20,2),
                        earnings_per_share_basic DECIMAL(10,4),
                        earnings_per_share_diluted DECIMAL(10,4),
                        weighted_average_shares_basic DECIMAL(20,0),
                        weighted_average_shares_diluted DECIMAL(20,0),
                        
                        -- LTM-specific metadata
                        calculation_method VARCHAR(50) NOT NULL, -- 'standard', 'quarterly_sum', etc.
                        calculation_inputs JSONB, -- Store details about which periods were used
                        ltm_calculation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        
                        -- Metadata
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        
                        -- Constraints - unique LTM calculation per quarter end date
                        UNIQUE(cik, base_quarter_end_date, period_type)
                    );
                """)
                
                # LTM Cash Flow Statements table - mirrors cash_flow_statements structure but stores 12-month rolling calculations  
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ltm_cash_flow_statements (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        company_id UUID NOT NULL REFERENCES companies(id),
                        cik VARCHAR(10) NOT NULL,
                        ticker VARCHAR(10) NOT NULL,
                        company_name TEXT NOT NULL,
                        
                        -- LTM Period information
                        period_end_date DATE NOT NULL,  -- End date of the 12-month period (standardized field name)
                        ltm_period_start_date DATE,         -- Start date of the 12-month period
                        base_quarter_end_date DATE NOT NULL, -- The quarter end date this LTM calculation is based on
                        fiscal_year INTEGER NOT NULL,
                        period_type VARCHAR(3) NOT NULL,  -- Q1, Q2, Q3, Q4 (standardized field name)
                        form_type VARCHAR(10),
                        
                        -- Currency and units
                        currency VARCHAR(3) DEFAULT 'USD',
                        units VARCHAR(10) DEFAULT 'USD',
                        
                        -- LTM Cash Flow Statement fields (same as cash_flow_statements table)
                        net_cash_from_operating_activities DECIMAL(20,2),
                        depreciation_and_amortization DECIMAL(20,2),
                        depreciation DECIMAL(20,2),
                        amortization DECIMAL(20,2),
                        stock_based_compensation DECIMAL(20,2),
                        changes_in_accounts_receivable DECIMAL(20,2),
                        changes_in_other_receivable DECIMAL(20,2),
                        changes_in_inventory DECIMAL(20,2),
                        changes_in_accounts_payable DECIMAL(20,2),
                        changes_in_other_operating_assets DECIMAL(20,2),
                        changes_in_other_operating_liabilities DECIMAL(20,2),
                        net_cash_from_investing_activities DECIMAL(20,2),
                        capital_expenditures DECIMAL(20,2),
                        acquisitions DECIMAL(20,2),
                        purchases_of_intangible_assets DECIMAL(20,2),
                        investments_purchased DECIMAL(20,2),
                        investments_sold DECIMAL(20,2),
                        divestitures DECIMAL(20,2),
                        net_cash_from_financing_activities DECIMAL(20,2),
                        dividends_paid DECIMAL(20,2),
                        share_repurchases DECIMAL(20,2),
                        proceeds_from_stock_issuance DECIMAL(20,2),
                        debt_issued DECIMAL(20,2),
                        debt_repaid DECIMAL(20,2),
                        net_change_in_cash DECIMAL(20,2),
                        cash_beginning_of_period DECIMAL(20,2),
                        cash_end_of_period DECIMAL(20,2),
                        
                        -- LTM-specific metadata
                        calculation_method VARCHAR(50) NOT NULL, -- 'standard', 'quarterly_sum', etc.
                        calculation_inputs JSONB, -- Store details about which periods were used
                        ltm_calculation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        
                        -- Metadata
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        
                        -- Constraints
                        UNIQUE(cik, base_quarter_end_date, period_type)
                    );
                """)
                
                # Create indexes for better performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_companies_ticker ON companies(ticker);
                    CREATE INDEX IF NOT EXISTS idx_companies_cik ON companies(cik);
                    
                    CREATE INDEX IF NOT EXISTS idx_income_statements_ticker ON income_statements(ticker);
                    CREATE INDEX IF NOT EXISTS idx_income_statements_cik ON income_statements(cik);
                    CREATE INDEX IF NOT EXISTS idx_income_statements_period ON income_statements(period_end_date, period_type);
                    CREATE INDEX IF NOT EXISTS idx_income_statements_fiscal_year ON income_statements(fiscal_year);
                    
                    CREATE INDEX IF NOT EXISTS idx_balance_sheets_ticker ON balance_sheets(ticker);
                    CREATE INDEX IF NOT EXISTS idx_balance_sheets_cik ON balance_sheets(cik);
                    CREATE INDEX IF NOT EXISTS idx_balance_sheets_period ON balance_sheets(period_end_date, period_type);
                    CREATE INDEX IF NOT EXISTS idx_balance_sheets_fiscal_year ON balance_sheets(fiscal_year);
                    
                    CREATE INDEX IF NOT EXISTS idx_cash_flow_statements_ticker ON cash_flow_statements(ticker);
                    CREATE INDEX IF NOT EXISTS idx_cash_flow_statements_cik ON cash_flow_statements(cik);
                    CREATE INDEX IF NOT EXISTS idx_cash_flow_statements_period ON cash_flow_statements(period_end_date, period_type);
                    CREATE INDEX IF NOT EXISTS idx_cash_flow_statements_fiscal_year ON cash_flow_statements(fiscal_year);
                    
                    CREATE INDEX IF NOT EXISTS idx_processing_metadata_ticker ON processing_metadata(ticker);
                    
                    CREATE INDEX IF NOT EXISTS idx_ratio_definitions_name ON ratio_definitions(name);
                    CREATE INDEX IF NOT EXISTS idx_ratio_definitions_company ON ratio_definitions(company_id);
                    CREATE INDEX IF NOT EXISTS idx_ratio_definitions_category ON ratio_definitions(category);
                    
                    CREATE INDEX IF NOT EXISTS idx_calculated_ratios_company ON calculated_ratios(company_id);
                    CREATE INDEX IF NOT EXISTS idx_calculated_ratios_ticker ON calculated_ratios(ticker);
                    CREATE INDEX IF NOT EXISTS idx_calculated_ratios_period ON calculated_ratios(period_end_date, period_type);
                    CREATE INDEX IF NOT EXISTS idx_calculated_ratios_definition ON calculated_ratios(ratio_definition_id);
                    CREATE INDEX IF NOT EXISTS idx_calculated_ratios_category ON calculated_ratios(ratio_category);
                    
                    CREATE INDEX IF NOT EXISTS idx_ltm_income_statements_ticker ON ltm_income_statements(ticker);
                    CREATE INDEX IF NOT EXISTS idx_ltm_income_statements_cik ON ltm_income_statements(cik);
                    CREATE INDEX IF NOT EXISTS idx_ltm_income_statements_period ON ltm_income_statements(period_end_date, period_type);
                    CREATE INDEX IF NOT EXISTS idx_ltm_income_statements_base_quarter ON ltm_income_statements(base_quarter_end_date);
                    CREATE INDEX IF NOT EXISTS idx_ltm_income_statements_fiscal_year ON ltm_income_statements(fiscal_year);
                    
                    CREATE INDEX IF NOT EXISTS idx_ltm_cash_flow_statements_ticker ON ltm_cash_flow_statements(ticker);
                    CREATE INDEX IF NOT EXISTS idx_ltm_cash_flow_statements_cik ON ltm_cash_flow_statements(cik);
                    CREATE INDEX IF NOT EXISTS idx_ltm_cash_flow_statements_period ON ltm_cash_flow_statements(period_end_date, period_type);
                    CREATE INDEX IF NOT EXISTS idx_ltm_cash_flow_statements_base_quarter ON ltm_cash_flow_statements(base_quarter_end_date);
                    CREATE INDEX IF NOT EXISTS idx_ltm_cash_flow_statements_fiscal_year ON ltm_cash_flow_statements(fiscal_year);
                """)
                
                self.connection.commit()
                logger.info("Database tables created successfully")
                
        except psycopg2.Error as e:
            self.connection.rollback()
            logger.error(f"Error creating tables: {e}")
            raise
    
    def insert_company(self, company: Company) -> uuid.UUID:
        """
        Insert or update company information
        
        Args:
            company: Company model instance
            
        Returns:
            Company ID (UUID)
        """
        try:
            with self.connection.cursor() as cursor:
                # Check if company already exists
                cursor.execute(
                    "SELECT id FROM companies WHERE cik = %s",
                    (company.cik,)
                )
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing company
                    cursor.execute("""
                        UPDATE companies SET
                            ticker = %s, name = %s, sic = %s, sic_description = %s,
                            ein = %s, description = %s, website = %s, investor_website = %s,
                            category = %s, fiscal_year_end = %s, state_of_incorporation = %s,
                            state_of_incorporation_description = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE cik = %s
                        RETURNING id
                    """, (
                        company.ticker, company.name, company.sic, company.sic_description,
                        company.ein, company.description, company.website, company.investor_website,
                        company.category, company.fiscal_year_end, company.state_of_incorporation,
                        company.state_of_incorporation_description, company.cik
                    ))
                    company_id = cursor.fetchone()[0]
                    logger.info(f"Updated company {company.ticker} (CIK: {company.cik})")
                else:
                    # Insert new company
                    cursor.execute("""
                        INSERT INTO companies (
                            cik, ticker, name, sic, sic_description, ein, description,
                            website, investor_website, category, fiscal_year_end,
                            state_of_incorporation, state_of_incorporation_description
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        company.cik, company.ticker, company.name, company.sic, company.sic_description,
                        company.ein, company.description, company.website, company.investor_website,
                        company.category, company.fiscal_year_end, company.state_of_incorporation,
                        company.state_of_incorporation_description
                    ))
                    company_id = cursor.fetchone()[0]
                    logger.info(f"Inserted new company {company.ticker} (CIK: {company.cik})")
                
                self.connection.commit()
                return company_id
                
        except psycopg2.Error as e:
            self.connection.rollback()
            logger.error(f"Error inserting company {company.ticker}: {e}")
            raise
    
    def _get_statement_table_name(self, statement_type: type) -> str:
        """Get the table name for a statement type"""
        if statement_type == IncomeStatement:
            return "income_statements"
        elif statement_type == BalanceSheet:
            return "balance_sheets"
        elif statement_type == CashFlowStatement:
            return "cash_flow_statements"
        else:
            raise ValueError(f"Unknown statement type: {statement_type}")
    
    def check_statement_exists(self, statement: Union[IncomeStatement, BalanceSheet, CashFlowStatement]) -> bool:
        """
        Check if a financial statement already exists
        
        Args:
            statement: Statement instance to check
            
        Returns:
            True if statement exists, False otherwise
        """
        try:
            table_name = self._get_statement_table_name(type(statement))
            # Validate table name to prevent SQL injection
            validated_table_name = self._validate_table_name(table_name)
            
            with self.connection.cursor() as cursor:
                # Use psycopg2.sql for safe table name injection
                from psycopg2 import sql
                query = sql.SQL("""
                    SELECT 1 FROM {table}
                    WHERE cik = %s AND period_end_date = %s AND period_type = %s AND period_length_months = %s
                """).format(table=sql.Identifier(validated_table_name))
                
                cursor.execute(query, (statement.cik, statement.period_end_date, statement.period_type, statement.period_length_months))
                
                return cursor.fetchone() is not None
                
        except psycopg2.Error as e:
            logger.error(f"Error checking statement existence: {e}")
            return False
        except ValueError as e:
            logger.error(f"Security error: {e}")
            return False
    
    def insert_statement(self, statement: Union[IncomeStatement, BalanceSheet, CashFlowStatement]) -> Optional[uuid.UUID]:
        """
        Insert financial statement data
        
        Args:
            statement: Statement model instance
            
        Returns:
            Statement ID (UUID) or None if duplicate
        """
        try:
            table_name = self._get_statement_table_name(type(statement))
            # Validate table name to prevent SQL injection
            validated_table_name = self._validate_table_name(table_name)
            
            with self.connection.cursor() as cursor:
                # Get all field values from the model, excluding auto-generated fields
                exclude_fields = {'id', 'created_at', 'updated_at'}
                fields = []
                values = []
                
                for field_name, field_info in statement.__fields__.items():
                    if field_name in exclude_fields:
                        continue
                    
                    value = getattr(statement, field_name)
                    # Convert UUID objects to strings for PostgreSQL compatibility
                    if isinstance(value, uuid.UUID):
                        value = str(value)
                    fields.append(field_name)
                    values.append(value)
                
                # Build the SQL query safely using psycopg2.sql
                from psycopg2 import sql
                placeholders = ', '.join(['%s'] * len(values))
                
                query = sql.SQL("""
                    INSERT INTO {table} ({fields})
                    VALUES ({placeholders})
                    RETURNING id
                """).format(
                    table=sql.Identifier(validated_table_name),
                    fields=sql.SQL(', ').join(map(sql.Identifier, fields)),
                    placeholders=sql.SQL(placeholders)
                )
                
                cursor.execute(query, values)
                statement_id = cursor.fetchone()[0]
                
                self.connection.commit()
                logger.info(f"Inserted {table_name} for {statement.ticker} - {statement.period_end_date}")
                return statement_id
                
        except psycopg2.IntegrityError as e:
            self.connection.rollback()
            if "duplicate key value" in str(e):
                logger.warning(f"Duplicate statement for {statement.ticker} - {statement.period_end_date}")
                return None
            else:
                logger.error(f"Integrity error inserting statement: {e}")
                raise
        except psycopg2.Error as e:
            self.connection.rollback()
            logger.error(f"Error inserting financial statement: {e}")
            raise
    
    def upsert_statement(self, statement: Union[IncomeStatement, BalanceSheet, CashFlowStatement]) -> str:
        """
        Insert or update financial statement data (UPSERT operation)
        
        Args:
            statement: Statement model instance
            
        Returns:
            'inserted', 'updated', or 'failed'
        """
        try:
            table_name = self._get_statement_table_name(type(statement))
            validated_table_name = self._validate_table_name(table_name)
            
            with self.connection.cursor() as cursor:
                # Check if statement exists
                from psycopg2 import sql
                check_query = sql.SQL("""
                    SELECT id FROM {table}
                    WHERE cik = %s AND period_end_date = %s AND period_type = %s AND period_length_months = %s
                """).format(table=sql.Identifier(validated_table_name))
                
                cursor.execute(check_query, (statement.cik, statement.period_end_date, statement.period_type, statement.period_length_months))
                existing_record = cursor.fetchone()
                
                if existing_record:
                    # Update existing record
                    existing_id = existing_record[0]
                    
                    # Get all field values from the model, excluding auto-generated fields
                    exclude_fields = {'id', 'created_at', 'updated_at'}
                    update_fields = []
                    values = []
                    
                    for field_name, field_info in statement.__fields__.items():
                        if field_name in exclude_fields:
                            continue
                        
                        value = getattr(statement, field_name)
                        # Convert UUID objects to strings for PostgreSQL compatibility
                        if isinstance(value, uuid.UUID):
                            value = str(value)
                        update_fields.append(field_name)
                        values.append(value)
                    
                    # Build the UPDATE query with COALESCE to preserve existing non-NULL values
                    # Only overwrite if the new value is not NULL
                    set_clauses = [
                        sql.SQL("{} = COALESCE(%s, {})").format(sql.Identifier(field), sql.Identifier(field)) 
                        for field in update_fields
                    ]
                    
                    update_query = sql.SQL("""
                        UPDATE {table} 
                        SET {fields}, updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                        RETURNING id
                    """).format(
                        table=sql.Identifier(validated_table_name),
                        fields=sql.SQL(', ').join(set_clauses)
                    )
                    
                    values.append(existing_id)
                    cursor.execute(update_query, values)
                    updated_id = cursor.fetchone()[0]
                    
                    self.connection.commit()
                    logger.info(f"Updated {table_name} for {statement.ticker} - {statement.period_end_date}")
                    return 'updated'
                    
                else:
                    # Insert new record
                    result_id = self.insert_statement(statement)
                    if result_id:
                        return 'inserted'
                    else:
                        return 'failed'
                        
        except psycopg2.Error as e:
            self.connection.rollback()
            logger.error(f"Error upserting financial statement: {e}")
            return 'failed'
        except ValueError as e:
            logger.error(f"Security error: {e}")
            return 'failed'
    
    def bulk_insert_statements(self, statements: List[Union[IncomeStatement, BalanceSheet, CashFlowStatement]], 
                             force_refresh: bool = False) -> Tuple[int, int]:
        """
        Bulk insert financial statements
        
        Args:
            statements: List of statement instances
            force_refresh: If True, overwrite existing records instead of skipping them
            
        Returns:
            Tuple of (inserted_count, skipped_count)
        """
        if not statements:
            return 0, 0
        
        inserted_count = 0
        skipped_count = 0
        updated_count = 0
        
        try:
            for statement in statements:
                if force_refresh:
                    # During force refresh, attempt to update existing records or insert new ones
                    result = self.upsert_statement(statement)
                    if result == 'inserted':
                        inserted_count += 1
                    elif result == 'updated':
                        updated_count += 1
                    else:
                        skipped_count += 1
                else:
                    # Normal behavior: check if statement already exists and skip
                    if self.check_statement_exists(statement):
                        skipped_count += 1
                        logger.debug(f"Skipping duplicate statement: {statement.ticker} - {statement.period_end_date}")
                        continue
                    
                    # Insert the statement
                    result = self.insert_statement(statement)
                    if result:
                        inserted_count += 1
                    else:
                        skipped_count += 1
            
            if force_refresh:
                logger.info(f"Force refresh completed: {inserted_count} inserted, {updated_count} updated, {skipped_count} skipped")
            else:
                logger.info(f"Bulk insert completed: {inserted_count} inserted, {skipped_count} skipped")
            return inserted_count, skipped_count
            
        except Exception as e:
            logger.error(f"Error in bulk insert: {e}")
            raise
    
    def update_processing_metadata(self, metadata: ProcessingMetadata):
        """
        Update processing metadata
        
        Args:
            metadata: ProcessingMetadata instance
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO processing_metadata (
                        ticker, cik, processing_status, periods_processed, periods_skipped,
                        error_message, last_processed
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, cik) DO UPDATE SET
                        processing_status = EXCLUDED.processing_status,
                        periods_processed = EXCLUDED.periods_processed,
                        periods_skipped = EXCLUDED.periods_skipped,
                        error_message = EXCLUDED.error_message,
                        last_processed = EXCLUDED.last_processed
                """, (
                    metadata.ticker, metadata.cik, metadata.processing_status,
                    metadata.periods_processed, metadata.periods_skipped,
                    metadata.error_message, metadata.last_processed
                ))
                
                self.connection.commit()
                logger.info(f"Updated processing metadata for {metadata.ticker}")
                
        except psycopg2.Error as e:
            self.connection.rollback()
            logger.error(f"Error updating processing metadata: {e}")
            raise
    
    def get_company_by_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get company information by ticker
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            Company data dictionary or None
        """
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT * FROM companies WHERE ticker = %s",
                    (self._sanitize_ticker(ticker),)
                )
                return cursor.fetchone()
                
        except psycopg2.Error as e:
            logger.error(f"Error getting company by ticker {ticker}: {e}")
            return None
    
    def is_company_data_fresh(self, ticker: str, hours: int = 24) -> bool:
        """
        Check if company data was updated within the specified number of hours
        
        Args:
            ticker: Company ticker symbol
            hours: Number of hours to check (default: 24)
            
        Returns:
            True if data is fresh (updated within specified hours), False otherwise
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT updated_at 
                    FROM companies 
                    WHERE ticker = %s 
                    AND updated_at > NOW() - INTERVAL '%s hours'
                """, (self._sanitize_ticker(ticker), hours))
                
                result = cursor.fetchone()
                is_fresh = result is not None
                
                if is_fresh:
                    logger.info(f"Company {ticker} data is fresh (updated within last {hours} hours)")
                else:
                    logger.info(f"Company {ticker} data is stale (needs refresh)")
                
                return is_fresh
                
        except psycopg2.Error as e:
            logger.error(f"Error checking data freshness for {ticker}: {e}")
            return False  # Assume stale on error to trigger refresh
    
    def update_company_timestamp(self, ticker: str) -> bool:
        """
        Update the updated_at timestamp for a company
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    UPDATE companies 
                    SET updated_at = CURRENT_TIMESTAMP 
                    WHERE ticker = %s
                    RETURNING updated_at
                """, (self._sanitize_ticker(ticker),))
                
                result = cursor.fetchone()
                if result:
                    logger.info(f"Updated timestamp for company {ticker} to {result[0]}")
                    self.connection.commit()
                    return True
                else:
                    logger.warning(f"No company found with ticker {ticker} to update timestamp")
                    return False
                
        except psycopg2.Error as e:
            logger.error(f"Error updating timestamp for {ticker}: {e}")
            self.connection.rollback()
            return False
    
    def reset_company_cache(self, ticker: str) -> bool:
        """
        Reset company cache by setting updated_at to a very old timestamp
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            True if reset was successful, False otherwise
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    UPDATE companies 
                    SET updated_at = '2020-01-01 00:00:00'::timestamp
                    WHERE ticker = %s
                    RETURNING updated_at
                """, (self._sanitize_ticker(ticker),))
                
                result = cursor.fetchone()
                if result:
                    logger.info(f"Reset cache for company {ticker} to {result[0]}")
                    self.connection.commit()
                    return True
                else:
                    logger.warning(f"No company found with ticker {ticker} to reset cache")
                    return False
                
        except psycopg2.Error as e:
            logger.error(f"Error resetting cache for {ticker}: {e}")
            self.connection.rollback()
            return False
    
    def force_refresh_company_data(self, ticker: str) -> bool:
        """
        Force refresh company data by removing all financial statements and marking for fresh download
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            True if refresh was successful, False otherwise
        """
        try:
            with self.connection.cursor() as cursor:
                # Get company info first
                company_info = self.get_company_by_ticker(ticker)
                if not company_info:
                    logger.warning(f"No company found with ticker {ticker} for force refresh")
                    return False
                
                cik = company_info['cik']
                
                # Delete all financial statements for this company (including LTM and ratio tables)
                # Note: Different tables use different identifier columns
                cik_tables = [
                    'income_statements', 
                    'balance_sheets', 
                    'cash_flow_statements',
                    'ltm_income_statements',
                    'ltm_cash_flow_statements'
                ]
                ticker_tables = [
                    'calculated_ratios'
                ]
                total_deleted = 0
                
                # Delete from tables that use CIK
                for table in cik_tables:
                    validated_table = self._validate_table_name(table)
                    from psycopg2 import sql
                    
                    delete_query = sql.SQL("DELETE FROM {table} WHERE cik = %s").format(
                        table=sql.Identifier(validated_table)
                    )
                    cursor.execute(delete_query, (cik,))
                    deleted_count = cursor.rowcount
                    total_deleted += deleted_count
                    logger.info(f"Deleted {deleted_count} records from {table} for {ticker} (using CIK)")
                
                # Delete from tables that use ticker
                for table in ticker_tables:
                    validated_table = self._validate_table_name(table)
                    from psycopg2 import sql
                    
                    delete_query = sql.SQL("DELETE FROM {table} WHERE ticker = %s").format(
                        table=sql.Identifier(validated_table)
                    )
                    cursor.execute(delete_query, (ticker,))
                    deleted_count = cursor.rowcount
                    total_deleted += deleted_count
                    logger.info(f"Deleted {deleted_count} records from {table} for {ticker} (using ticker)")
                
                # Commit the deletions first - this is the critical part
                logger.info(f"Committing deletion of {total_deleted} records for {ticker}")
                self.connection.commit()
                
                # Reset the company cache timestamp to force refresh (separate transaction)
                cursor.execute("""
                    UPDATE companies 
                    SET updated_at = '2020-01-01 00:00:00'::timestamp
                    WHERE cik = %s
                    RETURNING updated_at
                """, (cik,))
                
                result = cursor.fetchone()
                if result:
                    logger.info(f"Force refresh for {ticker}: deleted {total_deleted} records, reset cache timestamp to {result[0]}")
                    self.connection.commit()
                    return True
                else:
                    logger.error(f"Failed to reset cache timestamp for {ticker} - but deletions were committed")
                    self.connection.rollback()  # Only rollback the timestamp update
                    return True  # Still return True because deletions succeeded
                
        except psycopg2.Error as e:
            logger.error(f"Error force refreshing data for {ticker}: {e}")
            self.connection.rollback()
            return False
        except ValueError as e:
            logger.error(f"Security error: {e}")
            return False
    
    def get_income_statements(self, ticker: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get income statements for a company"""
        return self._get_statements("income_statements", ticker, limit)
    
    def get_balance_sheets(self, ticker: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get balance sheets for a company"""
        return self._get_statements("balance_sheets", ticker, limit)
    
    def get_cash_flow_statements(self, ticker: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get cash flow statements for a company"""
        return self._get_statements("cash_flow_statements", ticker, limit)
    
    def _get_statements(self, table_name: str, ticker: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get financial statements for a company from specified table
        
        Args:
            table_name: Name of the table to query
            ticker: Company ticker symbol
            limit: Maximum number of statements to return
            
        Returns:
            List of financial statement dictionaries
        """
        try:
            # Validate table name to prevent SQL injection
            validated_table_name = self._validate_table_name(table_name)
            
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                from psycopg2 import sql
                
                base_query = sql.SQL("""
                    SELECT * FROM {table}
                    WHERE ticker = %s 
                    ORDER BY period_end_date DESC, period_type
                """).format(table=sql.Identifier(validated_table_name))
                
                params = [self._sanitize_ticker(ticker)]
                
                if limit:
                    query = base_query + sql.SQL(" LIMIT %s")
                    params.append(limit)
                else:
                    query = base_query
                
                cursor.execute(query, params)
                return cursor.fetchall()
                
        except psycopg2.Error as e:
            logger.error(f"Error getting {table_name} for {ticker}: {e}")
            return []
    
    def get_calculated_ratios(self, ticker: str, ratio_name: Optional[str] = None, 
                            category: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get calculated ratios for a company
        
        Args:
            ticker: Company ticker symbol
            ratio_name: Specific ratio name (optional)
            category: Filter by ratio category (optional)
            limit: Maximum number of ratios to return
            
        Returns:
            List of calculated ratio dictionaries
        """
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                query = """
                    SELECT 
                        cr.ticker, cr.period_end_date, cr.period_type, cr.ratio_value,
                        cr.calculation_inputs, cr.data_source, cr.calculation_date,
                        cr.ratio_category, cr.fiscal_year, cr.fiscal_quarter,
                        cr.ratio_name, rd.description, rd.category, rd.formula
                    FROM calculated_ratios cr
                    JOIN ratio_definitions rd ON cr.ratio_definition_id = rd.id
                    WHERE cr.ticker = %s
                """
                params = [self._sanitize_ticker(ticker)]
                
                if ratio_name:
                    query += " AND cr.ratio_name = %s"
                    params.append(ratio_name)
                
                if category:
                    query += " AND rd.category = %s"
                    params.append(category)
                
                query += " ORDER BY rd.category, cr.ratio_name, cr.period_end_date DESC"
                
                if limit:
                    query += " LIMIT %s"
                    params.append(limit)
                
                cursor.execute(query, params)
                return cursor.fetchall()
                
        except Exception as e:
            logger.error(f"Error getting calculated ratios for {ticker}: {e}")
            return []
    
    def get_ratio_definitions_for_company(self, company_id: uuid.UUID) -> List[Dict[str, Any]]:
        """
        Get all ratio definitions for a company (hybrid: company-specific + global)
        
        Args:
            company_id: Company UUID
            
        Returns:
            List of ratio definition dictionaries
        """
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    WITH company_ratios AS (
                        SELECT *, 1 as priority FROM ratio_definitions 
                        WHERE company_id = %s AND is_active = true
                    ),
                    global_ratios AS (
                        SELECT *, 2 as priority FROM ratio_definitions 
                        WHERE company_id IS NULL AND is_active = true
                        AND name NOT IN (SELECT name FROM company_ratios)
                    )
                    SELECT * FROM company_ratios
                    UNION ALL
                    SELECT * FROM global_ratios
                    ORDER BY priority, category, name
                """, (company_id,))
                
                return cursor.fetchall()
                
        except Exception as e:
            logger.error(f"Error getting ratio definitions for company {company_id}: {e}")
            return []
    
    def insert_ltm_income_statement(self, ltm_data: Dict[str, Any]) -> Optional[uuid.UUID]:
        """
        Insert LTM income statement data
        
        Args:
            ltm_data: Dictionary with LTM income statement data
            
        Returns:
            LTM statement ID (UUID) or None if duplicate
        """
        try:
            with self.connection.cursor() as cursor:
                
                # Check if this LTM record already exists (unique per cik + period_end_date)
                cursor.execute("""
                    SELECT id, calculation_method, ltm_calculation_date 
                    FROM ltm_income_statements 
                    WHERE cik = %s AND base_quarter_end_date = %s
                """, (ltm_data['cik'], ltm_data['base_quarter_end_date']))
                
                existing = cursor.fetchone()
                if existing:
                    existing_id, existing_method, existing_date = existing
                    new_method = ltm_data.get('calculation_method', 'unknown')
                    
                    
                    # Determine if we should replace the existing record
                    should_replace = self._should_replace_ltm_record(existing_method, new_method, existing_date)
                    
                    if should_replace:
                        return self._update_ltm_income_statement(existing_id, ltm_data)
                    else:
                        return existing_id
                
                
                # Get all field values, excluding auto-generated fields
                exclude_fields = {'id', 'created_at', 'updated_at'}
                fields = []
                values = []
                
                for field_name, value in ltm_data.items():
                    if field_name in exclude_fields:
                        continue
                    
                    # Convert UUID objects to strings for PostgreSQL compatibility
                    if isinstance(value, uuid.UUID):
                        value = str(value)
                    # Convert dict objects to JSON strings for JSONB fields
                    elif isinstance(value, dict):
                        import json
                        value = json.dumps(value)
                    # Convert datetime objects to strings
                    elif hasattr(value, 'isoformat'):
                        value = value.isoformat()
                    
                    fields.append(field_name)
                    values.append(value)
                
                # Build the SQL query safely using psycopg2.sql
                from psycopg2 import sql
                placeholders = ', '.join(['%s'] * len(values))
                
                query = sql.SQL("""
                    INSERT INTO {table} ({fields})
                    VALUES ({placeholders})
                    RETURNING id
                """).format(
                    table=sql.Identifier('ltm_income_statements'),
                    fields=sql.SQL(', ').join(map(sql.Identifier, fields)),
                    placeholders=sql.SQL(placeholders)
                )
                
                cursor.execute(query, values)
                ltm_id = cursor.fetchone()[0]
                
                self.connection.commit()
                logger.info(f"Inserted LTM income statement for {ltm_data['ticker']} - {ltm_data['base_quarter_end_date']}")
                return ltm_id
                
        except psycopg2.IntegrityError as e:
            self.connection.rollback()
            if "duplicate key value" in str(e):
                logger.warning(f"Duplicate LTM income statement for {ltm_data['ticker']} - {ltm_data['base_quarter_end_date']}")
                return None
            else:
                logger.error(f"Integrity error inserting LTM income statement: {e}")
                raise
        except psycopg2.Error as e:
            self.connection.rollback()
            logger.error(f"Error inserting LTM income statement: {e}")
            raise
    
    def insert_ltm_cash_flow_statement(self, ltm_data: Dict[str, Any]) -> Optional[uuid.UUID]:
        """
        Insert LTM cash flow statement data
        
        Args:
            ltm_data: Dictionary with LTM cash flow statement data
            
        Returns:
            LTM statement ID (UUID) or None if duplicate
        """
        try:
            with self.connection.cursor() as cursor:
                
                # Check if this LTM record already exists (unique per cik + period_end_date)
                cursor.execute("""
                    SELECT id, calculation_method, ltm_calculation_date 
                    FROM ltm_cash_flow_statements 
                    WHERE cik = %s AND base_quarter_end_date = %s
                """, (ltm_data['cik'], ltm_data['base_quarter_end_date']))
                
                existing = cursor.fetchone()
                if existing:
                    existing_id, existing_method, existing_date = existing
                    new_method = ltm_data.get('calculation_method', 'unknown')
                    
                    
                    # Determine if we should replace the existing record
                    should_replace = self._should_replace_ltm_record(existing_method, new_method, existing_date)
                    
                    if should_replace:
                        return self._update_ltm_cash_flow_statement(existing_id, ltm_data)
                    else:
                        return existing_id
                
                # Get all field values, excluding auto-generated fields
                exclude_fields = {'id', 'created_at', 'updated_at'}
                fields = []
                values = []
                
                for field_name, value in ltm_data.items():
                    if field_name in exclude_fields:
                        continue
                    
                    # Convert UUID objects to strings for PostgreSQL compatibility
                    if isinstance(value, uuid.UUID):
                        value = str(value)
                    # Convert dict objects to JSON strings for JSONB fields
                    elif isinstance(value, dict):
                        import json
                        value = json.dumps(value)
                    # Convert datetime objects to strings
                    elif hasattr(value, 'isoformat'):
                        value = value.isoformat()
                    
                    fields.append(field_name)
                    values.append(value)
                
                # Build the SQL query safely using psycopg2.sql
                from psycopg2 import sql
                placeholders = ', '.join(['%s'] * len(values))
                
                query = sql.SQL("""
                    INSERT INTO {table} ({fields})
                    VALUES ({placeholders})
                    RETURNING id
                """).format(
                    table=sql.Identifier('ltm_cash_flow_statements'),
                    fields=sql.SQL(', ').join(map(sql.Identifier, fields)),
                    placeholders=sql.SQL(placeholders)
                )
                
                cursor.execute(query, values)
                ltm_id = cursor.fetchone()[0]
                
                self.connection.commit()
                logger.info(f"Inserted LTM cash flow statement for {ltm_data['ticker']} - {ltm_data['base_quarter_end_date']}")
                return ltm_id
                
        except psycopg2.IntegrityError as e:
            self.connection.rollback()
            if "duplicate key value" in str(e):
                logger.warning(f"Duplicate LTM cash flow statement for {ltm_data['ticker']} - {ltm_data['base_quarter_end_date']}")
                return None
            else:
                logger.error(f"Integrity error inserting LTM cash flow statement: {e}")
                raise
        except psycopg2.Error as e:
            self.connection.rollback()
            logger.error(f"Error inserting LTM cash flow statement: {e}")
            raise
    
    def get_ltm_income_statements(self, ticker: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get LTM income statements for a company"""
        return self._get_ltm_statements("ltm_income_statements", ticker, limit)
    
    def get_ltm_cash_flow_statements(self, ticker: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get LTM cash flow statements for a company"""
        return self._get_ltm_statements("ltm_cash_flow_statements", ticker, limit)
    
    def _get_ltm_statements(self, table_name: str, ticker: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get LTM financial statements for a company from specified table
        
        Args:
            table_name: Name of the LTM table to query
            ticker: Company ticker symbol
            limit: Maximum number of statements to return
            
        Returns:
            List of LTM financial statement dictionaries
        """
        try:
            # Validate table name to prevent SQL injection
            validated_table_name = self._validate_table_name(table_name)
            
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                from psycopg2 import sql
                
                base_query = sql.SQL("""
                    SELECT * FROM {table}
                    WHERE ticker = %s 
                    ORDER BY period_end_date DESC, period_type
                """).format(table=sql.Identifier(validated_table_name))
                
                params = [self._sanitize_ticker(ticker)]
                
                if limit:
                    query = base_query + sql.SQL(" LIMIT %s")
                    params.append(limit)
                else:
                    query = base_query
                
                cursor.execute(query, params)
                return cursor.fetchall()
                
        except psycopg2.Error as e:
            logger.error(f"Error getting {table_name} for {ticker}: {e}")
            return []
        except ValueError as e:
            logger.error(f"Security error: {e}")
            return []
    
    def bulk_insert_ltm_statements(self, ltm_statements: List[Dict[str, Any]], statement_type: str) -> Tuple[int, int]:
        """
        Bulk insert LTM statements
        
        Args:
            ltm_statements: List of LTM statement dictionaries
            statement_type: 'income_statement' or 'cash_flow'
            
        Returns:
            Tuple of (inserted_count, skipped_count)
        """
        
        if not ltm_statements:
            return 0, 0
        
        inserted_count = 0
        skipped_count = 0
        
        try:
            for i, statement in enumerate(ltm_statements):
                
                if statement_type == 'income_statement':
                    result = self.insert_ltm_income_statement(statement)
                elif statement_type == 'cash_flow':
                    result = self.insert_ltm_cash_flow_statement(statement)
                else:
                    logger.error(f"Unknown LTM statement type: {statement_type}")
                    skipped_count += 1
                    continue
                
                if result:
                    inserted_count += 1
                else:
                    skipped_count += 1
            
            logger.info(f"Bulk LTM insert completed: {inserted_count} inserted, {skipped_count} skipped")
            return inserted_count, skipped_count
            
        except Exception as e:
            logger.error(f"Error in bulk LTM insert: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _should_replace_ltm_record(self, existing_method: str, new_method: str, existing_date) -> bool:
        """
        Determine if we should replace an existing LTM record with a new one
        
        Priority ranking (higher priority wins):
        1. standard_full (best - has FY + same period data)
        2. enhanced_quarterly_sum (good - detailed quarterly calculation)  
        3. standard_fy_only (fallback - missing same period data)
        4. fy_direct (direct FY data converted to LTM)
        5. quarterly_sum (basic quarterly sum)
        6. unknown (lowest priority)
        """
        method_priority = {
            'standard_full': 6,
            'enhanced_quarterly_sum': 5,
            'standard_fy_only': 4,
            'fy_direct': 3,
            'quarterly_sum': 2,
            'unknown': 1
        }
        
        existing_priority = method_priority.get(existing_method, 1)
        new_priority = method_priority.get(new_method, 1)
        
        # Replace if new method has higher priority
        if new_priority > existing_priority:
            return True
        
        # If same priority, replace if calculation is more recent (within last day)
        if new_priority == existing_priority:
            from datetime import datetime, timedelta
            if existing_date:
                try:
                    if isinstance(existing_date, str):
                        existing_dt = datetime.fromisoformat(existing_date.replace('Z', '+00:00'))
                    else:
                        existing_dt = existing_date
                    
                    # If existing calculation is old (>24 hours), replace with new
                    if datetime.now() - existing_dt.replace(tzinfo=None) > timedelta(hours=24):
                        return True
                except Exception:
                    pass
        
        return False
    
    def _update_ltm_income_statement(self, existing_id: str, ltm_data: Dict[str, Any]) -> str:
        """Update existing LTM income statement record with new data"""
        try:
            with self.connection.cursor() as cursor:
                
                # Get all field values, excluding auto-generated fields
                exclude_fields = {'id', 'created_at', 'updated_at'}
                update_fields = []
                values = []
                
                for field_name, value in ltm_data.items():
                    if field_name in exclude_fields:
                        continue
                    
                    # Convert UUID objects to strings for PostgreSQL compatibility
                    if isinstance(value, uuid.UUID):
                        value = str(value)
                    # Convert dict objects to JSON strings for JSONB fields
                    elif isinstance(value, dict):
                        import json
                        value = json.dumps(value)
                    # Convert datetime objects to strings
                    elif hasattr(value, 'isoformat'):
                        value = value.isoformat()
                    
                    update_fields.append(field_name)
                    values.append(value)
                
                # Build the UPDATE query safely using psycopg2.sql
                from psycopg2 import sql
                set_clauses = [sql.SQL("{} = %s").format(sql.Identifier(field)) for field in update_fields]
                
                query = sql.SQL("""
                    UPDATE {table} 
                    SET {fields}, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    RETURNING id
                """).format(
                    table=sql.Identifier('ltm_income_statements'),
                    fields=sql.SQL(', ').join(set_clauses)
                )
                
                values.append(existing_id)
                cursor.execute(query, values)
                updated_id = cursor.fetchone()[0]
                
                self.connection.commit()
                logger.info(f"Updated LTM income statement for {ltm_data['ticker']} - {ltm_data['base_quarter_end_date']}")
                return str(updated_id)
                
        except Exception as e:
            logger.error(f"Error updating LTM income statement: {e}")
            raise
    
    def _update_ltm_cash_flow_statement(self, existing_id: str, ltm_data: Dict[str, Any]) -> str:
        """Update existing LTM cash flow statement record with new data"""
        try:
            with self.connection.cursor() as cursor:
                
                # Get all field values, excluding auto-generated fields
                exclude_fields = {'id', 'created_at', 'updated_at'}
                update_fields = []
                values = []
                
                for field_name, value in ltm_data.items():
                    if field_name in exclude_fields:
                        continue
                    
                    # Convert UUID objects to strings for PostgreSQL compatibility
                    if isinstance(value, uuid.UUID):
                        value = str(value)
                    # Convert dict objects to JSON strings for JSONB fields
                    elif isinstance(value, dict):
                        import json
                        value = json.dumps(value)
                    # Convert datetime objects to strings
                    elif hasattr(value, 'isoformat'):
                        value = value.isoformat()
                    
                    update_fields.append(field_name)
                    values.append(value)
                
                # Build the UPDATE query safely using psycopg2.sql
                from psycopg2 import sql
                set_clauses = [sql.SQL("{} = %s").format(sql.Identifier(field)) for field in update_fields]
                
                query = sql.SQL("""
                    UPDATE {table} 
                    SET {fields}, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    RETURNING id
                """).format(
                    table=sql.Identifier('ltm_cash_flow_statements'),
                    fields=sql.SQL(', ').join(set_clauses)
                )
                
                values.append(existing_id)
                cursor.execute(query, values)
                updated_id = cursor.fetchone()[0]
                
                self.connection.commit()
                logger.info(f"Updated LTM cash flow statement for {ltm_data['ticker']} - {ltm_data['base_quarter_end_date']}")
                return str(updated_id)
                
        except Exception as e:
            logger.error(f"Error updating LTM cash flow statement: {e}")
            raise
    
    def cleanup_ltm_duplicates(self, ticker: str = None) -> Dict[str, int]:
        """
        Clean up duplicate LTM records by keeping only the best record per (cik, base_quarter_end_date)
        
        Args:
            ticker: Optional ticker to limit cleanup to specific company
            
        Returns:
            Dict with cleanup statistics
        """
        try:
            stats = {
                'income_duplicates_removed': 0,
                'cash_flow_duplicates_removed': 0,
                'total_removed': 0
            }
            
            with self.connection.cursor() as cursor:
                # Build WHERE clause if ticker specified
                where_clause = ""
                params = []
                if ticker:
                    # Get company info to find CIK
                    company_info = self.get_company_by_ticker(ticker)
                    if company_info:
                        where_clause = "WHERE cik = %s"
                        params = [company_info['cik']]
                    else:
                        logger.warning(f"Company {ticker} not found for cleanup")
                        return stats
                
                # Clean up income statement duplicates
                cursor.execute(f"""
                    WITH ranked_records AS (
                        SELECT id, cik, base_quarter_end_date, calculation_method, ltm_calculation_date,
                               ROW_NUMBER() OVER (
                                   PARTITION BY cik, base_quarter_end_date 
                                   ORDER BY 
                                       CASE calculation_method
                                           WHEN 'standard_full' THEN 6
                                           WHEN 'enhanced_quarterly_sum' THEN 5
                                           WHEN 'standard_fy_only' THEN 4
                                           WHEN 'fy_direct' THEN 3
                                           WHEN 'quarterly_sum' THEN 2
                                           ELSE 1
                                       END DESC,
                                       ltm_calculation_date DESC
                               ) as rn
                        FROM ltm_income_statements
                        {where_clause}
                    )
                    DELETE FROM ltm_income_statements
                    WHERE id IN (
                        SELECT id FROM ranked_records WHERE rn > 1
                    )
                """, params)
                
                income_removed = cursor.rowcount
                stats['income_duplicates_removed'] = income_removed
                
                # Clean up cash flow statement duplicates
                cursor.execute(f"""
                    WITH ranked_records AS (
                        SELECT id, cik, base_quarter_end_date, calculation_method, ltm_calculation_date,
                               ROW_NUMBER() OVER (
                                   PARTITION BY cik, base_quarter_end_date 
                                   ORDER BY 
                                       CASE calculation_method
                                           WHEN 'standard_full' THEN 6
                                           WHEN 'enhanced_quarterly_sum' THEN 5
                                           WHEN 'standard_fy_only' THEN 4
                                           WHEN 'fy_direct' THEN 3
                                           WHEN 'quarterly_sum' THEN 2
                                           ELSE 1
                                       END DESC,
                                       ltm_calculation_date DESC
                               ) as rn
                        FROM ltm_cash_flow_statements
                        {where_clause}
                    )
                    DELETE FROM ltm_cash_flow_statements
                    WHERE id IN (
                        SELECT id FROM ranked_records WHERE rn > 1
                    )
                """, params)
                
                cash_flow_removed = cursor.rowcount
                stats['cash_flow_duplicates_removed'] = cash_flow_removed
                
                stats['total_removed'] = income_removed + cash_flow_removed
                
                self.connection.commit()
                logger.info(f"LTM cleanup completed: {stats['total_removed']} total duplicates removed")
                
                return stats
                
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error during LTM cleanup: {e}")
            raise

    def close(self):
        """Close database connection"""
        if self.connection and not self.connection.closed:
            self.connection.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
