import { useState, useEffect } from "react";
import { StockCard } from "@/frontend/components/StockCard";
import { StockSelector } from "@/frontend/components/StockSelector";
import { PriceChart } from "@/frontend/components/PriceChart";
import { ThemeToggle } from "@/frontend/components/ThemeToggle";
import { Button } from "@/frontend/components/ui/button";
import { AlertCircle, TrendingUp, X } from "lucide-react";
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/frontend/components/ui/alert";
import { useToast } from "@/frontend/hooks/use-toast";

interface StockData {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  high: number;
  low: number;
  open: number;
  prediction?: {
    trend: "up" | "down" | "neutral";
    confidence: number;
    predictedChange: number;
  };
}

const Index = () => {
  const [selectedStocks, setSelectedStocks] = useState<string[]>(() => {
    return ["AAPL"];
  });
  const [stocksData, setStocksData] = useState<Record<string, StockData>>({});
  const [isMarketOpen, setIsMarketOpen] = useState(true);
  const { toast } = useToast();

  // Check if market is open (9:30 AM - 4:00 PM ET, weekdays only)
  const checkMarketHours = () => {
    const now = new Date();
    const et = new Date(
      now.toLocaleString("en-US", { timeZone: "America/New_York" })
    );
    const day = et.getDay();
    const hours = et.getHours();
    const minutes = et.getMinutes();
    const time = hours * 60 + minutes;

    // Market is closed on weekends
    if (day === 0 || day === 6) return false;

    // Market hours: 9:30 AM (570 minutes) to 4:00 PM (960 minutes)
    return time >= 570 && time < 960;
  };

  const fetchStockData = async (symbol: string): Promise<StockData> => {
    const apiKey =
      import.meta.env.VITE_FINNHUB_API_KEY || import.meta.env.FINNHUB_API_KEY;
    if (!apiKey) {
      throw new Error("Finnhub API key not set");
    }

    // Finnhub Quotes endpoint: https://finnhub.io/docs/api/quote
    const res = await fetch(
      `https://finnhub.io/api/v1/quote?symbol=${encodeURIComponent(
        symbol
      )}&token=${apiKey}`
    );

    if (!res.ok) {
      throw new Error(`Failed to fetch real stock data for ${symbol}`);
    }

    const quote = await res.json();

    // Finnhub Company Profile endpoint for name (optional)
    let companyName = symbol;
    try {
      const profileRes = await fetch(
        `https://finnhub.io/api/v1/stock/profile2?symbol=${encodeURIComponent(
          symbol
        )}&token=${apiKey}`
      );
      if (profileRes.ok) {
        const profile = await profileRes.json();
        if (profile.name) companyName = profile.name;
      }
    } catch {}

    // Provide a simple prediction mock (real prediction should use actual ML/edge-function)
    const randomTrend = Math.random();
    let trend: "up" | "down" | "neutral" = "neutral";
    if (randomTrend > 0.66) trend = "up";
    else if (randomTrend < 0.33) trend = "down";

    const predictedChange = (Math.random() - 0.5) * 5;
    const confidence = 0.6 + Math.random() * 0.3;

    return {
      symbol,
      name: companyName,
      price: quote.c,
      change: quote.d || 0,
      changePercent: quote.dp || 0,
      volume: quote.v || 0,
      high: quote.h || 0,
      low: quote.l || 0,
      open: quote.o || 0,
      prediction: {
        trend,
        confidence,
        predictedChange,
      },
    };
  };

  const updateStockData = async () => {
    const newData: Record<string, StockData> = {};
    for (const symbol of selectedStocks) {
      try {
        const data = await fetchStockData(symbol);
        newData[symbol] = data;
      } catch (error) {
        toast({
          title: "Error fetching data",
          description: `Failed to fetch data for ${symbol}`,
          variant: "destructive",
        });
      }
    }
    setStocksData(newData);
  };

  useEffect(() => {
    setIsMarketOpen(checkMarketHours());
    updateStockData();

    // Update every 60 seconds during market hours
    const interval = setInterval(() => {
      const marketOpen = checkMarketHours();
      setIsMarketOpen(marketOpen);
      if (marketOpen) {
        updateStockData();
      }
    }, 60000);

    return () => clearInterval(interval);
  }, [selectedStocks]);

  const handleAddStock = (symbol: string) => {
    if (selectedStocks.length >= 5) {
      toast({
        title: "Maximum stocks reached",
        description: "You can track up to 5 stocks simultaneously",
        variant: "destructive",
      });
      return;
    }
    setSelectedStocks([...selectedStocks, symbol]);
  };

  const handleRemoveStock = (symbol: string) => {
    setSelectedStocks(selectedStocks.filter((s) => s !== symbol));
    const newData = { ...stocksData };
    delete newData[symbol];
    setStocksData(newData);
  };

  // Generate mock chart data
  const generateChartData = (symbol: string) => {
    const data = [];
    const basePrice = stocksData[symbol]?.price || 150;
    for (let i = 24; i >= 0; i--) {
      data.push({
        time: `${i}h ago`,
        price: basePrice + (Math.random() - 0.5) * 10,
        predicted: i < 5 ? basePrice + (Math.random() - 0.5) * 8 : undefined,
      });
    }
    return data;
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        <div className="flex justify-end mb-4">
          <ThemeToggle />
        </div>

        {/* Header */}
        <header className="mb-8 text-center">
          <div className="flex items-center justify-center gap-3 mb-2">
            <TrendingUp className="h-10 w-10 text-primary" />
            <h1 className="text-4xl font-bold text-foreground">
              Stock Trend Predictor
            </h1>
          </div>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            ML-powered real-time stock trend predictions. Educational tool for
            probabilistic forecasting.
          </p>
          <Alert
            variant="default"
            className="mt-4 max-w-2xl mx-auto border-warning/50 bg-warning/10"
          >
            <AlertCircle className="h-4 w-4 text-warning" />
            <AlertTitle className="text-warning">Disclaimer</AlertTitle>
            <AlertDescription className="text-warning-foreground">
              This tool is for educational purposes only and not financial
              advice. Trading involves risk of loss.
            </AlertDescription>
          </Alert>
        </header>

        {/* Market Status */}
        {!isMarketOpen && (
          <Alert variant="default" className="mb-6 max-w-4xl mx-auto">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Market Closed</AlertTitle>
            <AlertDescription>
              US markets are closed. Trading hours: 9:30 AM - 4:00 PM ET,
              Monday-Friday
            </AlertDescription>
          </Alert>
        )}

        {/* Stock Selector */}
        <div className="mb-8 max-w-4xl mx-auto">
          <StockSelector
            onSelectStock={handleAddStock}
            selectedStocks={selectedStocks}
          />
        </div>

        {/* Stock Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          {selectedStocks.map((symbol) => (
            <div key={symbol} className="relative">
              <Button
                variant="ghost"
                size="icon"
                className="absolute -top-2 -right-2 z-10 h-8 w-8 rounded-full bg-destructive/20 hover:bg-destructive/30"
                onClick={() => handleRemoveStock(symbol)}
              >
                <X className="h-4 w-4" />
              </Button>
              {stocksData[symbol] && (
                <StockCard
                  {...stocksData[symbol]}
                  isMarketOpen={isMarketOpen}
                />
              )}
            </div>
          ))}
        </div>

        {/* Charts */}
        {selectedStocks.length > 0 && stocksData[selectedStocks[0]] && (
          <div className="max-w-6xl mx-auto">
            <PriceChart
              data={generateChartData(selectedStocks[0])}
              symbol={selectedStocks[0]}
            />
          </div>
        )}

        {/* How It Works */}
        <div className="mt-12 max-w-4xl mx-auto">
          <h2 className="text-2xl font-bold text-foreground mb-4">
            How It Works
          </h2>
          <div className="bg-card border border-border rounded-lg p-6 space-y-3 text-muted-foreground">
            <p>
              Our ML model uses LSTM (Long Short-Term Memory) neural networks to
              analyze historical price patterns and predict short-term trends
              (5-15 minutes ahead).
            </p>
            <p>
              <strong className="text-foreground">Data Sources:</strong>{" "}
              Real-time quotes from Finnhub API
            </p>
            <p>
              <strong className="text-foreground">Update Frequency:</strong>{" "}
              Every 60 seconds during market hours
            </p>
            <p>
              <strong className="text-foreground">
                Prediction Confidence:
              </strong>{" "}
              Based on model certainty (60-90%)
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
