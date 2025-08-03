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
            
            with self.connection.cursor() as cursor:
                cursor.execute(f"""
                    SELECT 1 FROM {table_name}
                    WHERE cik = %s AND period_end_date = %s AND period_type = %s AND period_length_months = %s
                """, (statement.cik, statement.period_end_date, statement.period_type, statement.period_length_months))
                
                return cursor.fetchone() is not None
                
        except psycopg2.Error as e:
            logger.error(f"Error checking statement existence: {e}")
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
                
                # Build the SQL query dynamically
                placeholders = ', '.join(['%s'] * len(values))
                fields_str = ', '.join(fields)
                
                query = f"""
                    INSERT INTO {table_name} ({fields_str})
                    VALUES ({placeholders})
                    RETURNING id
                """
                
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
    
    def bulk_insert_statements(self, statements: List[Union[IncomeStatement, BalanceSheet, CashFlowStatement]]) -> Tuple[int, int]:
        """
        Bulk insert financial statements
        
        Args:
            statements: List of statement instances
            
        Returns:
            Tuple of (inserted_count, skipped_count)
        """
        if not statements:
            return 0, 0
        
        inserted_count = 0
        skipped_count = 0
        
        try:
            for statement in statements:
                # Check if statement already exists
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
                    (ticker.upper(),)
                )
                return cursor.fetchone()
                
        except psycopg2.Error as e:
            logger.error(f"Error getting company by ticker {ticker}: {e}")
            return None
    
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
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                query = f"""
                    SELECT * FROM {table_name}
                    WHERE ticker = %s 
                    ORDER BY period_end_date DESC, period_type
                """
                params = [ticker.upper()]
                
                if limit:
                    query += " LIMIT %s"
                    params.append(limit)
                
                cursor.execute(query, params)
                return cursor.fetchall()
                
        except psycopg2.Error as e:
            logger.error(f"Error getting {table_name} for {ticker}: {e}")
            return []
    
    def close(self):
        """Close database connection"""
        if self.connection and not self.connection.closed:
            self.connection.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
