import React, { useState, useEffect } from 'react';
import { TrendingUp, MapPin, Search, Filter, RefreshCw } from 'lucide-react';
import BackButton from './BackButton';

interface MarketRecord {
  state: string;
  district: string;
  market: string;
  commodity: string;
  variety: string;
  grade: string;
  arrival_date: string;
  min_price: string;
  max_price: string;
  modal_price: string;
}

interface ApiResponse {
  records: MarketRecord[];
  total: number;
  count: number;
}

const MarketTrends: React.FC = () => {
  const [marketData, setMarketData] = useState<MarketRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [userLocation, setUserLocation] = useState({
    state: '',
    district: ''
  });
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCommodity, setSelectedCommodity] = useState('');

  // Use backend API endpoint instead of direct API calls
  const fetchMarketData = async (state?: string, district?: string, commodity?: string, limit: number = 100) => {
    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams({
        limit: limit.toString()
      });
      
      if (state) {
        params.append('state', state);
      }
      if (district) {
        params.append('district', district);
      }
      if (commodity) {
        params.append('commodity', commodity);
      }

      const response = await fetch(`/api/market-data?${params.toString()}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.success) {
        setMarketData(data.data || []);
      } else {
        throw new Error(data.error || 'Failed to fetch market data');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch market data');
      console.error('Error fetching market data:', err);
      
      // Fallback to demo data on error
      setMarketData([
        {
          state: 'Maharashtra',
          district: 'Pune',
          market: 'Pune Market',
          commodity: 'Onion',
          variety: 'Medium',
          grade: 'FAQ',
          arrival_date: '2025-08-23',
          min_price: '2000',
          max_price: '2500',
          modal_price: '2200'
        },
        {
          state: 'Punjab',
          district: 'Ludhiana',
          market: 'Ludhiana Mandi',
          commodity: 'Wheat',
          variety: 'Common',
          grade: 'FAQ',
          arrival_date: '2025-08-23',
          min_price: '2800',
          max_price: '3200',
          modal_price: '3000'
        },
        {
          state: 'Gujarat',
          district: 'Ahmedabad',
          market: 'Ahmedabad Market',
          commodity: 'Cotton',
          variety: 'Medium Staple',
          grade: 'FAQ',
          arrival_date: '2025-08-23',
          min_price: '7500',
          max_price: '8200',
          modal_price: '7800'
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleLocationSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    fetchMarketData(userLocation.state, userLocation.district, selectedCommodity);
  };

  const getCurrentLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          // For demo purposes, we'll fetch all data since we can't reverse geocode without additional APIs
          fetchMarketData();
        },
        (error) => {
          console.error('Error getting location:', error);
          // Fallback to manual input
        }
      );
    }
  };

  useEffect(() => {
    // Load initial data
    fetchMarketData();
  }, []);

  const filteredData = marketData.filter(item => {
    const matchesSearch = searchTerm === '' || 
      item.commodity.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.market.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.state.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.district.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesCommodity = selectedCommodity === '' || 
      item.commodity === selectedCommodity;
    
    return matchesSearch && matchesCommodity;
  });

  const uniqueCommodities = Array.from(new Set(marketData.map(item => item.commodity)));

  const formatPrice = (price: string) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(parseFloat(price));
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header with Back Button */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <BackButton to="/dashboard" label="Back to Dashboard" />
            <div className="flex items-center gap-2">
              <TrendingUp className="text-green-600" size={32} />
              <h1 className="text-3xl font-bold text-gray-800">Market Trends</h1>
            </div>
          </div>
          <button
            onClick={() => fetchMarketData(userLocation.state, userLocation.district, selectedCommodity)}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            disabled={loading}
          >
            <RefreshCw className={loading ? 'animate-spin' : ''} size={20} />
            Refresh
          </button>
        </div>

        {/* Location Input */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <MapPin size={24} className="text-blue-600" />
            Set Your Location
          </h2>
          <form onSubmit={handleLocationSubmit} className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <input
              type="text"
              placeholder="Enter State"
              value={userLocation.state}
              onChange={(e) => setUserLocation(prev => ({ ...prev, state: e.target.value }))}
              className="px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <input
              type="text"
              placeholder="Enter District (Optional)"
              value={userLocation.district}
              onChange={(e) => setUserLocation(prev => ({ ...prev, district: e.target.value }))}
              className="px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <div className="flex gap-2">
              <button
                type="submit"
                className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                disabled={loading}
              >
                Get Market Data
              </button>
              <button
                type="button"
                onClick={getCurrentLocation}
                className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
              >
                <MapPin size={20} />
              </button>
            </div>
          </form>
        </div>

        {/* Filters */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Filter size={24} className="text-purple-600" />
            Filters
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
              <input
                type="text"
                placeholder="Search commodities, markets, locations..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <select
              value={selectedCommodity}
              onChange={(e) => setSelectedCommodity(e.target.value)}
              className="px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="">All Commodities</option>
              {uniqueCommodities.map(commodity => (
                <option key={commodity} value={commodity}>{commodity}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="bg-white rounded-lg shadow-md p-8 text-center">
            <RefreshCw className="animate-spin mx-auto mb-4 text-blue-600" size={48} />
            <p className="text-gray-600">Loading market data...</p>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-6 mb-6">
            <h3 className="text-red-800 font-semibold mb-2">Error Loading Data</h3>
            <p className="text-red-700">{error}</p>
          </div>
        )}

        {/* Market Data */}
        {!loading && !error && filteredData.length > 0 && (
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            <div className="px-6 py-4 bg-gray-50 border-b">
              <h2 className="text-xl font-semibold text-gray-800">
                Market Prices ({filteredData.length} results)
              </h2>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-100">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Location
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Commodity
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Variety
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Min Price
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Max Price
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Modal Price
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Date
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {filteredData.map((item, index) => (
                    <tr key={index} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm font-medium text-gray-900">{item.market}</div>
                        <div className="text-sm text-gray-500">{item.district}, {item.state}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm font-medium text-gray-900">{item.commodity}</div>
                        <div className="text-sm text-gray-500">Grade: {item.grade}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {item.variety}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-red-600">
                        {formatPrice(item.min_price)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-green-600">
                        {formatPrice(item.max_price)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-blue-600">
                        {formatPrice(item.modal_price)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {item.arrival_date}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* No Data State */}
        {!loading && !error && filteredData.length === 0 && marketData.length > 0 && (
          <div className="bg-white rounded-lg shadow-md p-8 text-center">
            <Search className="mx-auto mb-4 text-gray-400" size={48} />
            <h3 className="text-lg font-semibold text-gray-800 mb-2">No Results Found</h3>
            <p className="text-gray-600">Try adjusting your search filters or location.</p>
          </div>
        )}

        {!loading && !error && marketData.length === 0 && (
          <div className="bg-white rounded-lg shadow-md p-8 text-center">
            <TrendingUp className="mx-auto mb-4 text-gray-400" size={48} />
            <h3 className="text-lg font-semibold text-gray-800 mb-2">No Market Data Available</h3>
            <p className="text-gray-600">Please set your location to fetch market data.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default MarketTrends;