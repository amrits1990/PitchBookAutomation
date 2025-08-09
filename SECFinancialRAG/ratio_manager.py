"""
Ratio Manager - CRUD operations for ratio definitions
Handles creation, reading, updating, and deletion of ratio definitions with hybrid support
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import json

try:
    from .models import RatioDefinition
    from .database import FinancialDatabase
    from .virtual_fields import DEFAULT_RATIOS
except ImportError:
    from models import RatioDefinition
    from database import FinancialDatabase
    from virtual_fields import DEFAULT_RATIOS

logger = logging.getLogger(__name__)


class RatioManager:
    """
    Manages ratio definitions with hybrid global/company-specific support
    """
    
    def __init__(self):
        self.database = None
    
    def __enter__(self):
        self.database = FinancialDatabase()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.database:
            self.database.close()
    
    def initialize_default_ratios(self, created_by: str = "system") -> int:
        """
        Initialize default global ratio definitions from DEFAULT_RATIOS
        
        Args:
            created_by: User who is initializing the ratios
            
        Returns:
            Number of ratios created
        """
        logger.info("Initializing default ratio definitions")
        
        created_count = 0
        
        try:
            for ratio_name, ratio_config in DEFAULT_RATIOS.items():
                # Check if global ratio already exists
                existing = self.get_ratio_definition(ratio_name)
                if existing:
                    logger.debug(f"Ratio {ratio_name} already exists, skipping")
                    continue
                
                # Create the ratio definition
                ratio_def = RatioDefinition(
                    name=ratio_name,
                    company_id=None,  # Global ratio
                    formula=ratio_config['formula'],
                    description=ratio_config['description'],
                    category=ratio_config['category'],
                    created_by=created_by
                )
                
                if self.create_ratio_definition(ratio_def):
                    created_count += 1
                    logger.debug(f"Created default ratio: {ratio_name}")
            
            logger.info(f"Initialized {created_count} default ratio definitions")
            return created_count
            
        except Exception as e:
            logger.error(f"Error initializing default ratios: {e}")
            return created_count
    
    def create_ratio_definition(self, ratio_def: RatioDefinition) -> bool:
        """
        Create a new ratio definition
        
        Args:
            ratio_def: RatioDefinition model instance
            
        Returns:
            True if created successfully, False otherwise
        """
        try:
            with self.database.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO ratio_definitions (
                        name, company_id, formula, description, category, 
                        is_active, created_by, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    ratio_def.name, 
                    str(ratio_def.company_id) if ratio_def.company_id else None,
                    ratio_def.formula,
                    ratio_def.description, 
                    ratio_def.category, 
                    ratio_def.is_active,
                    ratio_def.created_by, 
                    ratio_def.created_at, 
                    ratio_def.updated_at
                ))
                
                ratio_id = cursor.fetchone()[0]
                self.database.connection.commit()
                
                scope = "company-specific" if ratio_def.company_id else "global"
                logger.info(f"Created {scope} ratio definition: {ratio_def.name} (ID: {ratio_id})")
                return True
                
        except Exception as e:
            self.database.connection.rollback()
            logger.error(f"Error creating ratio definition {ratio_def.name}: {e}")
            return False
    
    def get_ratio_definition(self, name: str, company_id: Optional[uuid.UUID] = None) -> Optional[Dict]:
        """
        Get a specific ratio definition (hybrid lookup)
        
        Args:
            name: Ratio name
            company_id: Company ID (None for global lookup)
            
        Returns:
            Ratio definition dictionary or None
        """
        try:
            with self.database.connection.cursor() as cursor:
                if company_id:
                    # Look for company-specific first, then global
                    cursor.execute("""
                        SELECT * FROM ratio_definitions 
                        WHERE name = %s AND (company_id = %s OR company_id IS NULL)
                        ORDER BY company_id NULLS LAST
                        LIMIT 1
                    """, (name, str(company_id) if company_id else None))
                else:
                    # Only global ratios
                    cursor.execute("""
                        SELECT * FROM ratio_definitions 
                        WHERE name = %s AND company_id IS NULL
                    """, (name,))
                
                result = cursor.fetchone()
                if result:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, result))
                return None
                
        except Exception as e:
            logger.error(f"Error getting ratio definition {name}: {e}")
            return None
    
    def get_all_ratio_definitions(self, company_id: Optional[uuid.UUID] = None, 
                                 category: Optional[str] = None, 
                                 active_only: bool = True) -> List[Dict]:
        """
        Get all ratio definitions for a company (hybrid: company-specific + global)
        
        Args:
            company_id: Company ID (None for global only)
            category: Filter by category
            active_only: Only return active ratios
            
        Returns:
            List of ratio definition dictionaries
        """
        try:
            with self.database.connection.cursor() as cursor:
                # Build base conditions
                conditions = []
                if category:
                    conditions.append("category = %s")
                
                if active_only:
                    conditions.append("is_active = true")
                
                condition_str = f" AND {' AND '.join(conditions)}" if conditions else ""
                
                # Build query based on parameters
                if company_id:
                    # Get both company-specific and global ratios (company-specific take precedence)
                    query = f"""
                        WITH company_ratios AS (
                            SELECT *, 1 as priority FROM ratio_definitions 
                            WHERE company_id = %s{condition_str}
                        ),
                        global_ratios AS (
                            SELECT *, 2 as priority FROM ratio_definitions 
                            WHERE company_id IS NULL{condition_str}
                            AND name NOT IN (SELECT name FROM company_ratios)
                        )
                        SELECT * FROM company_ratios
                        UNION ALL
                        SELECT * FROM global_ratios
                        ORDER BY priority, category, name
                    """
                    params = [str(company_id) if company_id else None]
                    if category:
                        params.extend([category, category])  # For both CTEs
                else:
                    # Only global ratios
                    query = f"""
                        SELECT *, 1 as priority FROM ratio_definitions 
                        WHERE company_id IS NULL{condition_str}
                        ORDER BY priority, category, name
                    """
                    params = []
                    
                # Add category parameter if specified
                if category:
                    params.append(category)
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                if results:
                    columns = [desc[0] for desc in cursor.description]
                    return [dict(zip(columns, result)) for result in results]
                return []
                
        except Exception as e:
            logger.error(f"Error getting ratio definitions: {e}")
            return []
    
    def update_ratio_definition(self, ratio_id: uuid.UUID, updates: Dict[str, Any]) -> bool:
        """
        Update a ratio definition
        
        Args:
            ratio_id: Ratio definition UUID
            updates: Dictionary of fields to update
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            if not updates:
                return True
            
            # Build update query
            update_fields = []
            params = []
            
            allowed_fields = ['formula', 'description', 'category', 'is_active']
            for field, value in updates.items():
                if field in allowed_fields:
                    update_fields.append(f"{field} = %s")
                    params.append(value)
            
            if not update_fields:
                logger.warning("No valid fields to update")
                return False
            
            # Always update the updated_at timestamp
            update_fields.append("updated_at = %s")
            params.append(datetime.utcnow())
            params.append(str(ratio_id) if ratio_id else None)
            
            with self.database.connection.cursor() as cursor:
                query = f"""
                    UPDATE ratio_definitions 
                    SET {', '.join(update_fields)}
                    WHERE id = %s
                    RETURNING name
                """
                
                cursor.execute(query, params)
                result = cursor.fetchone()
                
                if result:
                    self.database.connection.commit()
                    logger.info(f"Updated ratio definition: {result[0]}")
                    return True
                else:
                    logger.warning(f"Ratio definition {ratio_id} not found for update")
                    return False
                
        except Exception as e:
            self.database.connection.rollback()
            logger.error(f"Error updating ratio definition {ratio_id}: {e}")
            return False
    
    def delete_ratio_definition(self, ratio_id: uuid.UUID, soft_delete: bool = True) -> bool:
        """
        Delete a ratio definition
        
        Args:
            ratio_id: Ratio definition UUID
            soft_delete: If True, set is_active=False; if False, actually delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            with self.database.connection.cursor() as cursor:
                if soft_delete:
                    # Soft delete - just deactivate
                    cursor.execute("""
                        UPDATE ratio_definitions 
                        SET is_active = false, updated_at = %s
                        WHERE id = %s
                        RETURNING name
                    """, (datetime.utcnow(), str(ratio_id) if ratio_id else None))
                else:
                    # Hard delete - remove completely
                    # Note: This will fail if there are calculated ratios referencing this definition
                    cursor.execute("""
                        DELETE FROM ratio_definitions 
                        WHERE id = %s
                        RETURNING name
                    """, (str(ratio_id) if ratio_id else None,))
                
                result = cursor.fetchone()
                
                if result:
                    self.database.connection.commit()
                    action = "deactivated" if soft_delete else "deleted"
                    logger.info(f"Successfully {action} ratio definition: {result[0]}")
                    return True
                else:
                    logger.warning(f"Ratio definition {ratio_id} not found for deletion")
                    return False
                
        except Exception as e:
            self.database.connection.rollback()
            logger.error(f"Error deleting ratio definition {ratio_id}: {e}")
            return False
    
    def create_company_specific_ratio(self, ticker: str, name: str, formula: str, 
                                    description: str = None, category: str = None,
                                    created_by: str = None) -> bool:
        """
        Create a company-specific ratio definition
        
        Args:
            ticker: Company ticker
            name: Ratio name
            formula: Mathematical formula
            description: Ratio description
            category: Ratio category
            created_by: User creating the ratio
            
        Returns:
            True if created successfully, False otherwise
        """
        try:
            # Get company info
            company_info = self.database.get_company_by_ticker(ticker)
            if not company_info:
                logger.error(f"Company {ticker} not found")
                return False
            
            company_id = company_info['id']
            
            # Create the ratio definition
            ratio_def = RatioDefinition(
                name=name,
                company_id=uuid.UUID(company_id) if isinstance(company_id, str) else company_id,
                formula=formula,
                description=description,
                category=category,
                created_by=created_by
            )
            
            return self.create_ratio_definition(ratio_def)
            
        except Exception as e:
            logger.error(f"Error creating company-specific ratio for {ticker}: {e}")
            return False
    
    def export_ratio_definitions(self, company_id: Optional[uuid.UUID] = None) -> List[Dict]:
        """
        Export ratio definitions to a format suitable for backup/sharing
        
        Args:
            company_id: Company ID (None for global ratios only)
            
        Returns:
            List of ratio definitions in exportable format
        """
        try:
            ratios = self.get_all_ratio_definitions(company_id, active_only=False)
            
            # Convert to exportable format (remove IDs, timestamps)
            export_data = []
            for ratio in ratios:
                export_ratio = {
                    'name': ratio['name'],
                    'formula': ratio['formula'],
                    'description': ratio['description'],
                    'category': ratio['category'],
                    'is_global': ratio['company_id'] is None
                }
                export_data.append(export_ratio)
            
            logger.info(f"Exported {len(export_data)} ratio definitions")
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting ratio definitions: {e}")
            return []
    
    def import_ratio_definitions(self, ratio_data: List[Dict], company_id: Optional[uuid.UUID] = None,
                               created_by: str = "import", overwrite: bool = False) -> int:
        """
        Import ratio definitions from external format
        
        Args:
            ratio_data: List of ratio definition dictionaries
            company_id: Company ID for company-specific import (None for global)
            created_by: User performing the import
            overwrite: Whether to overwrite existing ratios
            
        Returns:
            Number of ratios successfully imported
        """
        imported_count = 0
        
        try:
            for ratio_config in ratio_data:
                # Check if ratio already exists
                existing = self.get_ratio_definition(ratio_config['name'], company_id)
                
                if existing and not overwrite:
                    logger.debug(f"Ratio {ratio_config['name']} already exists, skipping")
                    continue
                
                if existing and overwrite:
                    # Update existing ratio
                    updates = {
                        'formula': ratio_config['formula'],
                        'description': ratio_config.get('description'),
                        'category': ratio_config.get('category')
                    }
                    if self.update_ratio_definition(uuid.UUID(existing['id']), updates):
                        imported_count += 1
                else:
                    # Create new ratio
                    ratio_def = RatioDefinition(
                        name=ratio_config['name'],
                        company_id=str(company_id) if company_id else None,
                        formula=ratio_config['formula'],
                        description=ratio_config.get('description'),
                        category=ratio_config.get('category'),
                        created_by=created_by
                    )
                    
                    if self.create_ratio_definition(ratio_def):
                        imported_count += 1
            
            logger.info(f"Imported {imported_count} ratio definitions")
            return imported_count
            
        except Exception as e:
            logger.error(f"Error importing ratio definitions: {e}")
            return imported_count


# Convenience functions
def initialize_default_ratios(created_by: str = "system") -> int:
    """Initialize default ratios - convenience function"""
    with RatioManager() as manager:
        return manager.initialize_default_ratios(created_by)

def create_company_ratio(ticker: str, name: str, formula: str, 
                        description: str = None, category: str = None) -> bool:
    """Create company-specific ratio - convenience function"""
    with RatioManager() as manager:
        return manager.create_company_specific_ratio(ticker, name, formula, description, category)

def get_company_ratio_definitions(ticker: str) -> List[Dict]:
    """Get all ratio definitions for a company - convenience function"""
    with RatioManager() as manager:
        # Get company info first
        company_info = manager.database.get_company_by_ticker(ticker)
        if not company_info:
            return []
        
        return manager.get_all_ratio_definitions(uuid.UUID(company_info['id']))